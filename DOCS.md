# OpenProver internals

## Architecture

OpenProver is a plan-execute loop. `cli.py` parses args and sets up signal handling, then hands off to `Prover` which runs the main loop. The prover uses `LLMClient` (Claude CLI wrapper), `prompts.py` (all templates and parsing), and `TUI` (terminal interface). All state lives on disk in a run directory so sessions can be interrupted and resumed.

## Modules

### `cli.py`

Entry point. Parses arguments, creates a `Prover` and `TUI`, installs a SIGINT handler (first ctrl+c = graceful shutdown, second = force quit), runs the prover, prints cost summary on exit.

### `prover.py`

The `Prover` class owns the proving loop and all state.

**Init:** Creates or resumes a run directory (`runs/<slug>-<timestamp>/`). Loads or initializes the whiteboard. Resume is detected by checking for existing `WHITEBOARD.md` + `THEOREM.md`. Sets up `LLMClient` with an archive dir.

**Step flow** (`run` -> `_do_step`):

1. **Plan** (`_get_plan`): In interactive mode, asks the LLM for a structured plan (action + summary + reasoning) via JSON schema. Skipped in autonomous mode.

2. **Confirm**: In interactive mode, shows the plan via TUI. User can accept (Enter), give text feedback (triggers replan), or press `s/p/r/q/a` for summarize/pause/restart/quit/autonomous.

3. **Execute** (`_execute_step`): Sends full context (theorem, whiteboard, lemma index, plan) to the LLM. Response is free-form text with structured sections parsed out at the end (ACTION, WHITEBOARD, PROOF, etc). Free-form text is used here instead of JSON schema because long math content is more reliable that way.

4. **Post-process** based on the action:
   - `continue` / `explore_avenue` / `replan`: update whiteboard, keep going
   - `verify`: separate verification LLM call with a skeptical system prompt, result feeds back into next step
   - `literature_search`: Claude CLI call with WebSearch + WebFetch tools enabled, result feeds back
   - `prove_lemma`: extract and store the lemma in `lemmas/<slug>/`
   - `declare_proof`: verify the full proof, save to `PROOF.md` if it passes
   - `declare_stuck`: only allowed after 80% of steps used
   - `check_counterexample`: explore whether the theorem might be false

5. **Save**: Write step input/output to `steps/step_NNN/`

**Other methods:**
- `_build_lemma_index`: scans `lemmas/` to build a summary string for context
- `_write_discussion`: generates a discussion/analysis doc at session end
- `_do_summary`: on-demand progress summary (triggered by `s`)
- `_reset`: restart proof search (clears whiteboard, lemmas, step counter)

### `llm.py`

`LLMClient` wraps the Claude CLI. Two modes:

**Non-streaming** (no `stream_callback`):
```
claude -p --model <model> --system-prompt <...> --output-format json --tools "" [--json-schema <...>]
```
Returns parsed JSON with `result`, `cost`, `duration_ms`, `raw`.

**Streaming** (with `stream_callback`):
```
claude -p --model <model> --system-prompt <...> --output-format stream-json --verbose --include-partial-messages --tools ""
```
Uses `Popen` + `readline()` instead of the line iterator (the iterator's read-ahead buffer defeats real-time streaming). Parses NDJSON lines, dispatches `content_block_delta` text to the callback, captures the final `result` message.

**Web search:** When `web_search=True`, replaces `--tools ""` with `--permission-mode bypassPermissions --allowedTools WebSearch WebFetch`.

**Archiving:** Every call (success or failure) is saved to `archive/calls/call_NNN.json` with the full prompt, system prompt, schema, response, timing, and errors.

**Gotchas:**
- `--json-schema` puts structured output in `raw["structured_output"]`, not `raw["result"]`
- `--tools ""` disables all tools (pure reasoning mode)
- Cost tracking uses `total_cost_usd` from the Claude CLI response

### `prompts.py`

All prompt templates, JSON schemas, and output parsing.

**Actions**: the set of possible step actions:
```python
ACTIONS = [
    "continue", "explore_avenue", "prove_lemma", "verify",
    "check_counterexample", "literature_search", "replan",
    "declare_proof", "declare_stuck",
]
```
`ACTIONS_NO_SEARCH` excludes `literature_search` for `--isolation` mode.

**System prompt** (`SYSTEM_PROMPT`): instructs the LLM to behave as a research mathematician. Key rules:
- Whiteboard style: terse, dense, LaTeX math
- Literature search: max 2-3 per session, don't repeat
- Proof standards: must be complete and rigorous for `declare_proof`
- Anti-giving-up: must use 80%+ of steps before `declare_stuck`

**Plan schema** (`PLAN_SCHEMA`): JSON schema for the planning step. Returns `{action, summary, reasoning, target}`. The action enum is adjusted dynamically based on available actions.

**Step output format**: uses section-based parsing instead of JSON schema. Sections:
```
ACTION: <action>
SUMMARY: <one-line>
WHITEBOARD: ... END_WHITEBOARD
PROOF: ... END_PROOF
VERIFY_TARGET: <target>
VERIFY_CONTENT: ... END_VERIFY_CONTENT
SEARCH_QUERY: <query>
LEMMA_NAME: <name>
LEMMA_SOURCE: <citation>
LEMMA_CONTENT: ... END_LEMMA_CONTENT
```

`parse_step_output()` extracts these from the LLM response text.

**Other prompt formatters:**
- `format_verify_prompt` + `VERIFY_SYSTEM_PROMPT`: independent verification (skeptical checker)
- `format_literature_search_prompt` + `LITERATURE_SEARCH_SYSTEM_PROMPT`: web search
- `format_discussion_prompt`: end-of-session analysis
- `format_summary_prompt`: on-demand progress summary

### `tui.py`

Full-screen terminal UI using ANSI escape codes and scroll regions.

**Layout:** Rows 1-4 are a fixed header (box with theorem name, step counter, key hints). Row 5+ is the scrolling content region.

**Views:** `main` (log + streaming trace), `whiteboard`, `help`, `step_detail`.

**Key handling:** A background thread (`_bg_loop`) reads stdin in cbreak mode using `select` + `os.read`. Keys like `t/w/?/a` are handled directly on the background thread during streaming, so you can toggle trace while the LLM is running. Other keys go to `_key_queue` for the main thread to consume.

**Streaming:** Shows a braille spinner while waiting for the first token, then switches to streaming trace text (toggleable with `t`). Spinner updates at ~12fps.

**Confirmation UI:** Two-option selector (accept / give feedback) with a text input buffer. Up/down arrows browse step history; Enter on a history entry opens a detail view.

**Thread safety:** All stdout writes go through `_write_lock`. Background thread handles key processing and spinner; main thread handles log entries and redraws.

## Run directory

```
runs/<slug>-<timestamp>/
  THEOREM.md              - immutable copy of input
  WHITEBOARD.md           - latest whiteboard state (enables resume)
  PROOF.md                - written only if proof found + verified
  DISCUSSION.md           - post-session analysis
  lemmas/<slug>/LEMMA.md  - statement + metadata
  lemmas/<slug>/PROOF.md  - proof content
  steps/step_001/input.md - whiteboard at start of step
  steps/step_001/output.md - whiteboard after step
  steps/step_001/action.json - {step, action, summary}
  archive/calls/call_001.json - full LLM call record
```

Resume: if `--run-dir` points to a directory with `WHITEBOARD.md` + `THEOREM.md`, the prover picks up from the last step. Step count is inferred from the number of `step_NNN` directories.

## Verification

Verification uses a separate LLM call with a different system prompt (`VERIFY_SYSTEM_PROMPT`) that tells the model to be skeptical and not fill in gaps. The verifier hasn't seen the reasoning that produced the proof; it checks cold.

Two paths:
1. **Step verification** (`action=verify`): verify a specific argument mid-proof. Result feeds back as context for the next step.
2. **Final proof verification** (`action=declare_proof`): the full proof is verified before acceptance. If verification fails, the prover keeps working.

The verifier must end its response with `VERDICT: CORRECT` or `VERDICT: INCORRECT`.

## Adding a new action

1. Add the action name to `ACTIONS` in `prompts.py`
2. Add any new output sections to `_STEP_OUTPUT_BASE` or create a new section block
3. Update `parse_step_output()` to extract the new fields
4. Handle the action in `Prover._do_step()` (the post-processing block)
5. Optionally add a color in `ACTION_STYLE` in `tui.py`
