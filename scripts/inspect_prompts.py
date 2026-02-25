#!/usr/bin/env python3
"""Browse LLM prompts and outputs from an OpenProver run directory."""

import argparse
import json
import os
import signal
import shutil
import sys
import termios
import textwrap
import tty
from pathlib import Path

# ANSI codes
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
RED = "\033[38;5;174m"
GREEN = "\033[38;5;114m"
YELLOW = "\033[38;5;222m"
BLUE = "\033[38;5;75m"
CYAN = "\033[38;5;116m"
GRAY = "\033[38;5;245m"

HEADER_ROWS = 3  # header + controls + separator


def parse_args():
    p = argparse.ArgumentParser(description="Browse prompts/outputs from an OpenProver run")
    p.add_argument("run_dir", nargs="?", help="Run directory (default: most recent in runs/)")
    return p.parse_args()


def find_latest_run() -> Path:
    """Find the most recently modified run directory."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("Error: no runs/ directory found", file=sys.stderr)
        sys.exit(1)
    dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not dirs:
        print("Error: no runs found in runs/", file=sys.stderr)
        sys.exit(1)
    return max(dirs, key=lambda d: d.stat().st_mtime)


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None


def format_tokens(data: dict) -> str:
    """Extract token info from archive JSON for the header."""
    resp = data.get("response") or {}
    usage = resp.get("usage", {})
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_create = usage.get("cache_creation_input_tokens", 0)
    parts = []
    if inp:
        parts.append(f"in:{inp}")
    if out:
        parts.append(f"out:{out}")
    if cache_read:
        parts.append(f"cache_r:{cache_read}")
    if cache_create:
        parts.append(f"cache_w:{cache_create}")
    return " ".join(parts)


def format_cost(data: dict) -> str:
    resp = data.get("response") or {}
    cost = resp.get("total_cost_usd", 0.0)
    if cost:
        return f"${cost:.4f}"
    return ""


def format_duration(data: dict) -> str:
    ms = data.get("elapsed_ms", 0)
    if ms:
        return f"{ms / 1000:.1f}s"
    return ""


def make_pages(data: dict, step: int | str, role: str, label: str) -> list[dict]:
    """Create prompt and output pages from an archive JSON dict."""
    pages = []
    model = data.get("model", "")
    meta_parts = [p for p in [model, format_duration(data), format_tokens(data), format_cost(data)] if p]
    meta = " | ".join(meta_parts)

    # Prompt page
    sys_prompt = data.get("system_prompt", "")
    user_prompt = data.get("prompt", "")
    prompt_parts = []
    if sys_prompt:
        prompt_parts.append(("dim", "── System Prompt ──\n\n" + sys_prompt))
    if sys_prompt and user_prompt:
        prompt_parts.append(("normal", "\n\n── User Prompt ──\n\n" + user_prompt))
    elif user_prompt:
        prompt_parts.append(("normal", user_prompt))

    pages.append({
        "type": "prompt",
        "step": step,
        "label": f"{label} Prompt",
        "segments": prompt_parts,
        "thinking": "",
        "metadata": meta,
    })

    # Output page
    error = data.get("error")
    result = data.get("result_text", "")
    thinking = data.get("thinking", "")
    response = data.get("response")

    out_segments = []
    if error:
        out_segments.append(("red", f"Error: {error}"))
        if result:
            out_segments.append(("normal", "\n\n" + result))
    elif result:
        out_segments.append(("normal", result))
    elif response is None and not error:
        out_segments.append(("dim", "(in progress — waiting for LLM response)"))
    else:
        out_segments.append(("dim", "(no output)"))

    pages.append({
        "type": "error" if error else "output",
        "step": step,
        "label": f"{label} Output",
        "segments": out_segments,
        "thinking": thinking,
        "metadata": meta,
    })

    return pages


def load_pages(run_dir: Path) -> list[dict]:
    """Load all pages from a run directory."""
    pages = []
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        print(f"Error: no steps/ directory in {run_dir}", file=sys.stderr)
        sys.exit(1)

    step_dirs = sorted(
        [d for d in steps_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
    )

    for step_dir in step_dirs:
        step_num = int(step_dir.name.removeprefix("step_"))

        # Planner call
        planner = load_json(step_dir / "planner_call.json")
        if planner:
            pages.extend(make_pages(planner, step_num, "planner", "Planner"))

        # Retries
        retry_idx = 1
        while True:
            retry_path = step_dir / f"planner_call_retry_{retry_idx}.json"
            retry_data = load_json(retry_path)
            if not retry_data:
                break
            pages.extend(make_pages(retry_data, step_num, "retry", f"Retry {retry_idx}"))
            retry_idx += 1

        # Workers
        workers_dir = step_dir / "workers"
        if workers_dir.exists():
            # Worker calls (spawn action)
            worker_idx = 0
            while True:
                worker_path = workers_dir / f"worker_{worker_idx}_call.json"
                worker_data = load_json(worker_path)
                if not worker_data:
                    break
                pages.extend(make_pages(worker_data, step_num, "worker", f"Worker {worker_idx}"))
                worker_idx += 1

            # Search call (literature_search action)
            search_data = load_json(workers_dir / "search_call.json")
            if search_data:
                pages.extend(make_pages(search_data, step_num, "search", "Search"))

    # Discussion call at run root
    discussion = load_json(run_dir / "discussion_call.json")
    if discussion:
        pages.extend(make_pages(discussion, "end", "discussion", "Discussion"))

    return pages


class InspectTUI:
    def __init__(self, pages: list[dict], run_dir: Path):
        self.pages = pages
        self.run_dir = run_dir
        self.page_idx = 0
        self.scroll_offset = 0
        self.trace_visible = False
        self.rows = 0
        self.cols = 0
        self._old_termios = None
        self._old_sigwinch = None

    def run(self):
        if not self.pages:
            print("No pages to display.")
            return

        size = shutil.get_terminal_size()
        self.cols, self.rows = size.columns, size.lines

        try:
            self._old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except (termios.error, OSError):
            self._old_termios = None

        # Alt screen, hide cursor, enable mouse
        sys.stdout.write("\033[?1049h\033[2J\033[?25l\033[?1000h\033[?1006h")
        sys.stdout.flush()

        self._old_sigwinch = signal.getsignal(signal.SIGWINCH)
        signal.signal(signal.SIGWINCH, self._on_resize)

        try:
            self._draw()
            while True:
                key = self._read_key()
                if key in ("q", "\x1b"):
                    break
                elif key == "right":
                    if self.page_idx < len(self.pages) - 1:
                        self.page_idx += 1
                        self.scroll_offset = 0
                        self._draw()
                elif key == "left":
                    if self.page_idx > 0:
                        self.page_idx -= 1
                        self.scroll_offset = 0
                        self._draw()
                elif key == "t":
                    self.trace_visible = not self.trace_visible
                    self._draw()
                elif key in ("down", "j", "scroll_down"):
                    self.scroll_offset += 3
                    self._draw()
                elif key in ("up", "k", "scroll_up"):
                    self.scroll_offset = max(0, self.scroll_offset - 3)
                    self._draw()
                elif key == "pgdn":
                    content_h = self.rows - HEADER_ROWS
                    self.scroll_offset += content_h // 2
                    self._draw()
                elif key == "pgup":
                    content_h = self.rows - HEADER_ROWS
                    self.scroll_offset = max(0, self.scroll_offset - content_h // 2)
                    self._draw()
                elif key == " ":
                    # Space scrolls down a page
                    content_h = self.rows - HEADER_ROWS
                    self.scroll_offset += content_h // 2
                    self._draw()
                elif key == "g":
                    self.page_idx = 0
                    self.scroll_offset = 0
                    self._draw()
                elif key == "G":
                    self.page_idx = len(self.pages) - 1
                    self.scroll_offset = 0
                    self._draw()
        finally:
            self._cleanup()

    def _on_resize(self, signum, frame):
        size = shutil.get_terminal_size()
        self.cols, self.rows = size.columns, size.lines
        self._draw()

    def _cleanup(self):
        sys.stdout.write("\033[?1000l\033[?1006l\033[?1049l\033[?25h")
        sys.stdout.flush()
        if self._old_termios:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)
            except (termios.error, OSError):
                pass
        if self._old_sigwinch is not None:
            signal.signal(signal.SIGWINCH, self._old_sigwinch)

    def _read_key(self) -> str:
        """Read a single keypress, handling escape sequences and mouse events."""
        ch = os.read(sys.stdin.fileno(), 1).decode("utf-8", errors="replace")
        if ch == "\x1b":
            import select as sel
            if sel.select([sys.stdin], [], [], 0.05)[0]:
                ch2 = os.read(sys.stdin.fileno(), 1).decode("utf-8", errors="replace")
                if ch2 == "[":
                    ch3 = os.read(sys.stdin.fileno(), 1).decode("utf-8", errors="replace")
                    if ch3 == "A":
                        return "up"
                    elif ch3 == "B":
                        return "down"
                    elif ch3 == "C":
                        return "right"
                    elif ch3 == "D":
                        return "left"
                    elif ch3 == "5":
                        os.read(sys.stdin.fileno(), 1)  # consume ~
                        return "pgup"
                    elif ch3 == "6":
                        os.read(sys.stdin.fileno(), 1)  # consume ~
                        return "pgdn"
                    elif ch3 == "<":
                        # SGR mouse event: \033[<button;x;yM or \033[<button;x;ym
                        buf = ""
                        while True:
                            c = os.read(sys.stdin.fileno(), 1).decode("utf-8", errors="replace")
                            if c in ("M", "m"):
                                break
                            buf += c
                        parts = buf.split(";")
                        if len(parts) >= 1:
                            btn = int(parts[0])
                            if btn == 64:
                                return "scroll_up"
                            elif btn == 65:
                                return "scroll_down"
                        return ""  # ignore other mouse events
                    else:
                        # Consume rest of unknown sequence
                        while sel.select([sys.stdin], [], [], 0.01)[0]:
                            os.read(sys.stdin.fileno(), 1)
                return "\x1b"
            return "\x1b"
        return ch

    def _render_lines(self, page: dict) -> list[str]:
        """Build the display lines for a page, with ANSI codes."""
        lines = []
        width = self.cols

        # Show thinking first if trace is visible
        if self.trace_visible and page.get("thinking"):
            thinking = page["thinking"]
            lines.append(f"{GRAY}── Reasoning Trace ──{RESET}")
            lines.append("")
            for raw_line in thinking.splitlines():
                wrapped = textwrap.wrap(raw_line, width - 1) if raw_line.strip() else [""]
                for wl in wrapped:
                    lines.append(f"{GRAY}{wl}{RESET}")
            lines.append("")
            lines.append(f"{GRAY}── End Trace ──{RESET}")
            lines.append("")

        # Segments
        style_map = {
            "normal": "",
            "dim": DIM,
            "red": RED,
        }
        for style, text in page.get("segments", []):
            prefix = style_map.get(style, "")
            suffix = RESET if prefix else ""
            for raw_line in text.splitlines():
                wrapped = textwrap.wrap(raw_line, width - 1) if raw_line.strip() else [""]
                for wl in wrapped:
                    lines.append(f"{prefix}{wl}{suffix}")

        return lines

    def _draw(self):
        page = self.pages[self.page_idx]
        content_h = self.rows - HEADER_ROWS
        lines = self._render_lines(page)

        # Clamp scroll
        max_scroll = max(0, len(lines) - content_h)
        self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))

        buf = []
        # Clear screen
        buf.append("\033[2J\033[H")

        # Header line 1: step | label | page position | metadata
        step = page["step"]
        step_str = f"Step {step}" if isinstance(step, int) else str(step).capitalize()
        label = page["label"]
        page_type = page["type"]
        type_color = RED if page_type == "error" else BLUE if page_type == "prompt" else GREEN
        pos = f"[{self.page_idx + 1}/{len(self.pages)}]"
        meta = page.get("metadata", "")
        trace_indicator = f"{YELLOW}[trace on]{RESET}" if self.trace_visible else f"{DIM}[trace off]{RESET}"

        header = f" {BOLD}{step_str}{RESET} {DIM}│{RESET} {type_color}{label}{RESET} {DIM}│{RESET} {pos}"
        if trace_indicator:
            header += f" {trace_indicator}"
        if meta:
            header += f" {DIM}│ {meta}{RESET}"
        buf.append(header)

        # Header line 2: controls
        scroll_info = ""
        if len(lines) > content_h:
            pct = int(self.scroll_offset / max(1, max_scroll) * 100)
            scroll_info = f" {DIM}scroll:{pct}%{RESET}"
        controls = f" {DIM}←/→ pages  ↑/↓/scroll  t trace  g/G first/last  q quit{scroll_info}{RESET}"
        buf.append(controls)

        # Separator
        buf.append(f" {DIM}{'─' * (self.cols - 2)}{RESET}")

        # Content area
        visible = lines[self.scroll_offset:self.scroll_offset + content_h]
        for line in visible:
            buf.append(f" {line}")

        # Pad remaining lines
        for _ in range(content_h - len(visible)):
            buf.append("")

        sys.stdout.write("\n".join(buf))
        sys.stdout.flush()


def main():
    args = parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run()

    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {run_dir.name}...", end="", flush=True)
    pages = load_pages(run_dir)
    print(f" {len(pages)} pages")

    if not pages:
        print("No LLM calls found.")
        sys.exit(0)

    tui = InspectTUI(pages, run_dir)
    tui.run()


if __name__ == "__main__":
    main()
