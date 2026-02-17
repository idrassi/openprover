"""Core proving loop for OpenProver."""

import json
import re
from datetime import datetime
from pathlib import Path

from . import prompts
from .llm import LLMClient
from .tui import TUI


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:50].strip("-")


class Prover:
    def __init__(self, theorem_path: str | None, model: str, max_steps: int,
                 autonomous: bool, verbose: bool, tui: TUI,
                 isolation: bool = False, run_dir: str | None = None):
        self.model = model
        self.max_steps = max_steps
        self.autonomous = autonomous
        self.verbose = verbose
        self.isolation = isolation
        self.tui = tui
        self.shutting_down = False
        self.step_num = 0
        self.verification_result = ""
        self.search_result = ""
        self.proof_text = ""
        self.resumed = False

        # Pick action list and schemas based on isolation mode
        if isolation:
            self.actions = prompts.ACTIONS_NO_SEARCH
        else:
            self.actions = prompts.ACTIONS
        self.plan_schema = prompts._make_plan_schema(self.actions)

        # Set up working directory
        if run_dir:
            self.work_dir = Path(run_dir)
        else:
            theorem_text = Path(theorem_path).read_text()
            first_line = theorem_text.strip().split("\n")[0][:40]
            slug = slugify(first_line) or "theorem"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.work_dir = Path("runs") / f"{slug}-{timestamp}"

        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "lemmas").mkdir(exist_ok=True)
        (self.work_dir / "steps").mkdir(exist_ok=True)
        (self.work_dir / "archive" / "calls").mkdir(parents=True, exist_ok=True)

        # Check for resume (existing run with WHITEBOARD.md)
        whiteboard_path = self.work_dir / "WHITEBOARD.md"
        theorem_file = self.work_dir / "THEOREM.md"
        if whiteboard_path.exists() and theorem_file.exists():
            # Resume from existing run
            self.whiteboard = whiteboard_path.read_text()
            self.theorem_text = theorem_file.read_text()
            steps_dir = self.work_dir / "steps"
            existing = [d for d in steps_dir.iterdir()
                        if d.is_dir() and d.name.startswith("step_")]
            self.step_num = len(existing)
            self.resumed = True
        else:
            if not theorem_path:
                raise SystemExit(
                    f"Error: no WHITEBOARD.md in {self.work_dir} — "
                    "provide a theorem file to start a new run"
                )
            self.theorem_text = Path(theorem_path).read_text()
            (self.work_dir / "THEOREM.md").write_text(self.theorem_text)
            self.whiteboard = prompts.format_initial_whiteboard(self.theorem_text)
            whiteboard_path.write_text(self.whiteboard)

        # LLM client
        self.llm = LLMClient(model, self.work_dir / "archive" / "calls")

        # Derive theorem name for header — collapse full text into one line
        lines = self.theorem_text.strip().splitlines()
        parts = []
        for line in lines:
            stripped = line.lstrip("#").strip()
            if stripped:
                parts.append(stripped)
        self.theorem_name = " ".join(parts)

    def run(self):
        # Set up TUI
        self.tui.setup(
            theorem_name=self.theorem_name,
            work_dir=str(self.work_dir),
            step_num=self.step_num,
            max_steps=self.max_steps,
        )
        self.tui.autonomous = self.autonomous
        self.tui.whiteboard = self.whiteboard

        if self.resumed:
            self.tui.log(
                f"Resuming from step {self.step_num}/{self.max_steps}",
                color="cyan",
            )

        paused = False
        try:
            while self.step_num < self.max_steps and not self.shutting_down:
                self.step_num += 1
                action = self._do_step()
                if action == "stop":
                    break
                if action == "pause":
                    paused = True
                    self.tui.log("Paused.", color="yellow")
                    break
        except KeyboardInterrupt:
            self.shutting_down = True

        if not paused and self.tui.step_entries:
            self._write_discussion()

    def _do_step(self) -> str:
        """Execute one step. Returns: 'continue', 'stop', 'pause'."""
        # Sync autonomous state from TUI (user may have toggled with 'a')
        self.autonomous = self.tui.autonomous

        lemma_index = self._build_lemma_index()

        # Check for autonomous mode key actions
        if self.autonomous:
            action = self.tui.get_pending_action()
            if action == "quit":
                self.shutting_down = True
                return "stop"
            if action == "pause":
                return "pause"
            if action == "summarize":
                self._do_summary()

        # Save input whiteboard for this step
        step_dir = self.work_dir / "steps" / f"step_{self.step_num:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "input.md").write_text(self.whiteboard)

        if not self.autonomous:
            # Interactive: plan first, then get approval
            plan = self._get_plan(lemma_index)
            if plan is None:
                if self.shutting_down:
                    return "stop"
                self.tui.log("Step failed — retrying...", color="red")
                return "continue"

            # Check if user toggled autonomous during planning
            if self.tui.autonomous:
                self.autonomous = True
                self.tui.log("→ autonomous mode", dim=True)
            else:
                self.tui.show_proposal(plan)

                # Confirmation loop
                while True:
                    resp = self.tui.get_confirmation()
                    if resp == "":
                        break  # accept plan
                    if resp == "q":
                        self.shutting_down = True
                        return "stop"
                    if resp == "p":
                        return "pause"
                    if resp == "s":
                        self._do_summary()
                        self.tui.show_proposal(plan)
                        continue
                    if resp == "a":
                        self.autonomous = True
                        self.tui.log("→ autonomous mode", dim=True)
                        break
                    # User gave feedback — replan
                    self.tui.log("Replanning...", color="yellow")
                    plan = self._get_plan(lemma_index, human_feedback=resp)
                    if plan is None:
                        if self.shutting_down:
                            return "stop"
                        self.tui.log("Step failed — retrying...", color="red")
                        return "continue"
                    self.tui.show_proposal(plan)

            result = self._execute_step(lemma_index, plan=plan)
        else:
            # Autonomous: single combined call
            result = self._execute_step(lemma_index)

        if result is None:
            self.tui.log("Step failed — retrying...", color="red")
            return "continue"

        action = result.get("action", "continue")

        # Update whiteboard (only if step produced one)
        if result.get("whiteboard"):
            self.whiteboard = result["whiteboard"]
            (self.work_dir / "WHITEBOARD.md").write_text(self.whiteboard)
            self.tui.whiteboard = self.whiteboard

        step_idx = len(self.tui.step_entries)
        self.tui.step_complete(
            self.step_num, self.max_steps, action, result.get("summary", ""),
        )

        # Handle new/updated lemmas
        self._process_lemmas(result)

        # Save step data
        self._save_step(result)

        # Handle verification
        if action == "verify" and result.get("verify_content"):
            self._do_verification(result["verify_target"], result["verify_content"])
            self.tui.update_step_detail(step_idx,
                f"Verify: {result.get('verify_target', '')}\n\n"
                f"{self.verification_result}")

        # Handle literature search
        if action == "literature_search" and result.get("search_query"):
            if self.isolation:
                self.tui.log("Literature search skipped (isolation mode)", color="yellow")
            else:
                self._do_literature_search(result["search_query"])
            self.tui.update_step_detail(step_idx,
                f"Query: {result['search_query']}\n\n{self.search_result}")

        # Handle terminal actions
        if action == "declare_proof":
            proof = result.get("proof", "")
            if not proof:
                self.tui.log("✗ Proof rejected: No proof text provided", color="red")
                return "continue"
            self.tui.update_step_detail(step_idx, f"Proof:\n\n{proof}")
            passed = self._verify_proof(proof)
            if passed:
                self.proof_text = proof
                (self.work_dir / "PROOF.md").write_text(proof)
                self.tui.log("✓ Proof found!", color="green", bold=True)
                self.tui.log(f"→ {self.work_dir / 'PROOF.md'}", dim=True)
                return "stop"
            else:
                self.tui.log("✗ Proof rejected: Verification failed — continuing", color="red")
                return "continue"

        if action == "declare_stuck":
            if self.step_num < self.max_steps * 0.8:
                self.tui.log(
                    f"Not giving up yet — only {self.step_num}/{self.max_steps} steps used",
                    color="yellow",
                )
                return "continue"
            self.tui.log("Stuck — no more ideas.", color="yellow")
            return "stop"

        return "continue"

    def _get_plan(self, lemma_index: str, human_feedback: str = "") -> dict | None:
        prompt = prompts.format_plan_prompt(
            theorem=self.theorem_text,
            whiteboard=self.whiteboard,
            lemma_index=lemma_index,
            step_num=self.step_num,
            max_steps=self.max_steps,
            actions=self.actions,
            human_feedback=human_feedback,
            verification_result=self.verification_result,
            search_result=self.search_result,
        )
        self.tui.stream_start("planning")
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.SYSTEM_PROMPT,
                json_schema=self.plan_schema,
                label=f"plan_step_{self.step_num}",
                stream_callback=self.tui.stream_text,
            )
        except RuntimeError as e:
            self.tui.stream_end()
            self.tui.log(f"Error: {e}", color="red")
            return None
        self.tui.stream_end()

        try:
            return json.loads(resp["result"])
        except json.JSONDecodeError:
            self.tui.log("Warning: failed to parse plan, using fallback", color="yellow")
            return {"action": "continue", "summary": "Continue work",
                    "reasoning": "Fallback due to parse error"}

    def _execute_step(self, lemma_index: str, plan: dict | None = None) -> dict | None:
        prompt = prompts.format_step_prompt(
            theorem=self.theorem_text,
            whiteboard=self.whiteboard,
            lemma_index=lemma_index,
            step_num=self.step_num,
            max_steps=self.max_steps,
            actions=self.actions,
            plan=plan,
            verification_result=self.verification_result,
            search_result=self.search_result,
        )
        self.verification_result = ""
        self.search_result = ""

        action_label = plan["action"] if plan else "working"
        self.tui.stream_start(action_label)
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.SYSTEM_PROMPT,
                label=f"step_{self.step_num}",
                stream_callback=self.tui.stream_text,
            )
        except RuntimeError as e:
            self.tui.stream_end()
            self.tui.log(f"Error: {e}", color="red")
            return None
        self.tui.stream_end()

        result = prompts.parse_step_output(resp["result"])
        if result is None:
            self.tui.log("Warning: failed to parse step output", color="yellow")
        return result

    @staticmethod
    def _parse_verdict(text: str) -> bool:
        for line in reversed(text.strip().splitlines()):
            line = line.strip().upper()
            if line == "VERDICT: CORRECT":
                return True
            if line == "VERDICT: INCORRECT":
                return False
        return False

    def _verify_proof(self, proof: str) -> bool:
        self.tui.log("Verifying: declared proof", color="yellow")
        prompt = prompts.format_verify_prompt(self.theorem_text, proof)
        self.tui.stream_start("verifying proof")
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.VERIFY_SYSTEM_PROMPT,
                label=f"verify_proof_step_{self.step_num}",
                stream_callback=self.tui.stream_text,
            )
            self.tui.stream_end()
            self.verification_result = resp["result"]
            passed = self._parse_verdict(resp["result"])
            if passed:
                self.tui.log("✓ Passed", color="green")
            else:
                self.tui.log("✗ Issues found", color="red")
            return passed
        except RuntimeError as e:
            self.tui.stream_end()
            self.tui.log(f"Verification error: {e}", color="red")
            self.verification_result = f"Verification failed due to error: {e}"
            return False

    def _do_verification(self, target: str, content: str):
        self.tui.log(f"Verifying: {target}", color="yellow")
        prompt = prompts.format_verify_prompt(target, content)
        self.tui.stream_start("verifying")
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.VERIFY_SYSTEM_PROMPT,
                label=f"verify_step_{self.step_num}",
                stream_callback=self.tui.stream_text,
            )
            self.tui.stream_end()
            self.verification_result = resp["result"]
            passed = self._parse_verdict(resp["result"])
            if passed:
                self.tui.log("✓ Passed", color="green")
            else:
                self.tui.log("✗ Issues found", color="red")
        except RuntimeError as e:
            self.tui.stream_end()
            self.tui.log(f"Verification error: {e}", color="red")
            self.verification_result = f"Verification failed due to error: {e}"

    def _do_literature_search(self, query: str):
        self.tui.log(f"Searching: {query}", color="magenta")
        prompt = prompts.format_literature_search_prompt(query, self.theorem_text)
        self.tui.stream_start("searching")
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.LITERATURE_SEARCH_SYSTEM_PROMPT,
                label=f"search_step_{self.step_num}",
                web_search=True,
                stream_callback=self.tui.stream_text,
            )
            self.tui.stream_end()
            self.search_result = resp["result"]
            # Show a brief summary of findings
            shown = 0
            for line in self.search_result.splitlines():
                line = line.strip().lstrip("#").strip()
                if not line:
                    continue
                if len(line) > 120:
                    line = line[:117] + "..."
                self.tui.log(f"  {line}", dim=True)
                shown += 1
                if shown >= 3:
                    break
        except RuntimeError as e:
            self.tui.stream_end()
            self.tui.log(f"Search error: {e}", color="red")
            self.search_result = f"Literature search failed: {e}"

    def _process_lemmas(self, result: dict):
        if result.get("lemma_name") and result.get("lemma_content"):
            name = slugify(result["lemma_name"]) or "unnamed-lemma"
            lemma_dir = self.work_dir / "lemmas" / name
            lemma_dir.mkdir(parents=True, exist_ok=True)

            source = result.get("lemma_source", "")
            content = result["lemma_content"]

            header = f"# {result['lemma_name']}\n\n"
            if source:
                header += f"**Source**: {source}\n\n"
            (lemma_dir / "LEMMA.md").write_text(header + content + "\n")

            proof_text = ""
            if source:
                proof_text = f"**Source**: {source}\n\n"
            proof_text += content
            (lemma_dir / "PROOF.md").write_text(proof_text + "\n")

            if source:
                self.tui.log(f"+ lemma: {result['lemma_name']} [{source}]", color="green")
            else:
                self.tui.log(f"+ lemma: {result['lemma_name']}", color="green")

    def _save_step(self, result: dict):
        step_dir = self.work_dir / "steps" / f"step_{self.step_num:03d}"
        (step_dir / "output.md").write_text(result.get("whiteboard", ""))
        (step_dir / "action.json").write_text(json.dumps({
            "step": self.step_num,
            "action": result.get("action"),
            "summary": result.get("summary"),
        }, indent=2))

    def _build_lemma_index(self) -> str:
        lemmas_dir = self.work_dir / "lemmas"
        if not lemmas_dir.exists():
            return ""
        entries = []
        for d in sorted(lemmas_dir.iterdir()):
            if not d.is_dir():
                continue
            lemma_file = d / "LEMMA.md"
            if not lemma_file.exists():
                continue
            has_proof = (d / "PROOF.md").exists()
            status = "proven" if has_proof else "unproven"
            content = lemma_file.read_text().strip().split("\n")
            first_line = content[0] if content else d.name
            entries.append(f"- **{d.name}** [{status}]: {first_line}")
        return "\n".join(entries)

    def _write_discussion(self):
        if self.shutting_down:
            self.tui.log("Interrupted — writing discussion... (ctrl+c again to exit immediately)", color="yellow")
        lemma_index = self._build_lemma_index()
        prompt = prompts.format_discussion_prompt(
            theorem=self.theorem_text,
            whiteboard=self.whiteboard,
            lemma_index=lemma_index,
            steps_taken=self.step_num,
            max_steps=self.max_steps,
            proof=self.proof_text,
        )
        self.tui.stream_start("writing discussion")
        try:
            resp = self.llm.call(
                prompt=prompt,
                system_prompt=prompts.SYSTEM_PROMPT,
                label="discussion",
                stream_callback=self.tui.stream_text,
            )
            self.tui.stream_end()
            (self.work_dir / "DISCUSSION.md").write_text(resp["result"])
            self.tui.log(f"→ {self.work_dir / 'DISCUSSION.md'}", dim=True)
        except RuntimeError as e:
            self.tui.stream_end()
            self.tui.log(f"Error generating discussion: {e}", color="red")
            (self.work_dir / "DISCUSSION.md").write_text(
                f"# Discussion\n\nSession ended after {self.step_num} steps.\n\n"
                f"## Final Whiteboard\n\n{self.whiteboard}\n"
            )
            self.tui.log(f"→ {self.work_dir / 'DISCUSSION.md'}", dim=True)

    def _do_summary(self):
        lemma_index = self._build_lemma_index()
        prompt = prompts.format_summary_prompt(
            theorem=self.theorem_text,
            whiteboard=self.whiteboard,
            lemma_index=lemma_index,
            step_num=self.step_num,
            max_steps=self.max_steps,
        )
        self.tui.stream_start("summarizing")
        try:
            self.llm.call(
                prompt=prompt,
                system_prompt=prompts.SYSTEM_PROMPT,
                label=f"summary_step_{self.step_num}",
                stream_callback=self.tui.stream_text,
            )
            self.tui.stream_end()
        except RuntimeError as e:
            self.tui.stream_end()
            self.tui.log(f"Summary error: {e}", color="red")

    def request_shutdown(self):
        self.shutting_down = True
        self.tui.interrupt()
