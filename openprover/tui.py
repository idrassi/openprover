"""Terminal UI for OpenProver — ANSI scroll regions, fixed header, inline input."""

import os
import queue
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty

from openprover import __version__

# 256-color palette
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
WHITE = "\033[97m"
BLUE = "\033[38;5;75m"
GREEN = "\033[38;5;114m"
YELLOW = "\033[38;5;222m"
RED = "\033[38;5;174m"
MAGENTA = "\033[38;5;183m"
CYAN = "\033[38;5;116m"

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

ACTION_STYLE = {
    "continue": CYAN,
    "explore_avenue": BLUE,
    "prove_lemma": GREEN,
    "verify": YELLOW,
    "check_counterexample": YELLOW,
    "literature_search": MAGENTA,
    "replan": YELLOW,
    "declare_proof": GREEN,
    "declare_stuck": RED,
}

HELP_TEXT = f"""\
  {BOLD}Controls{RESET}

  {DIM}Instant keys (work any time):{RESET}
    t           toggle reasoning trace
    w           toggle whiteboard view
    a           toggle autonomous mode
    ?           this help
    esc/enter   dismiss overlay

  {DIM}When confirming a plan:{RESET}
    {DIM}up/down{RESET}     browse step history
    tab         switch accept / feedback
    enter       confirm or view step detail
    esc         close detail / deselect
    s           summarize progress
    p           pause (resume with --run-dir)
    q           quit

  {DIM}In autonomous mode all keys are instant.{RESET}
  {DIM}Press ? or enter to dismiss.{RESET}
"""

COLOR_MAP = {
    "red": RED, "green": GREEN, "blue": BLUE,
    "yellow": YELLOW, "magenta": MAGENTA, "cyan": CYAN,
}

HEADER_ROWS = 4


class _LogEntry:
    """A line in the log. step_idx >= 0 marks completed-step lines."""
    __slots__ = ("text", "step_idx")

    def __init__(self, text: str, step_idx: int = -1):
        self.text = text
        self.step_idx = step_idx


class TUI:
    def __init__(self):
        self.rows = 0
        self.cols = 0
        self.log_lines: list[_LogEntry] = []
        self.trace_buf: list[str] = []
        self.trace_visible = True
        self.view = "main"
        self.whiteboard = ""
        self.pending_action: str | None = None
        self.streaming = False
        self.autonomous = False
        self._old_termios = None
        self._active = False
        self.theorem_name = ""
        self.step_num = 0
        self.max_steps = 0
        self._old_sigwinch = None
        # Background key reader
        self._key_queue: queue.Queue[str] = queue.Queue()
        self._bg_thread: threading.Thread | None = None
        self._bg_stop = False
        # Spinner state
        self._spinner_label = ""
        self._spinner_tick = 0
        self._spinner_time = 0.0
        # Step history — each entry: {action, summary, step_num, detail}
        self.step_entries: list[dict] = []
        self._nav_step = -1  # -1 = options focused, 0..N-1 = step index
        self._step_detail_text = ""
        self._step_detail_title = ""
        # Confirmation state
        self._confirming = False
        self._confirm_selected = 0
        self._confirm_buf: list[str] = []
        # Thread safety for stdout
        self._write_lock = threading.Lock()

    _content_start = HEADER_ROWS + 1

    def setup(self, theorem_name: str, work_dir: str,
              step_num: int = 0, max_steps: int = 50):
        self.theorem_name = theorem_name
        self.step_num = step_num
        self.max_steps = max_steps
        size = shutil.get_terminal_size()
        self.cols, self.rows = size.columns, size.lines

        try:
            self._old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except (termios.error, OSError):
            self._old_termios = None

        with self._write_lock:
            self._write_raw('\033[?1049h\033[2J')
            self._draw_header()
            self._write_raw(f'\033[{self._content_start};{self.rows}r')
            sys.stdout.flush()
        self._write(f'\033[{self._content_start};1H\033[?25l')
        self._active = True

        self._old_sigwinch = signal.getsignal(signal.SIGWINCH)
        signal.signal(signal.SIGWINCH, self._on_resize)

        self._bg_stop = False
        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_thread.start()

    def cleanup(self):
        if not self._active:
            return
        self._active = False
        self._bg_stop = True
        if self._bg_thread:
            self._bg_thread.join(timeout=0.2)
            self._bg_thread = None
        self._write('\033[r\033[?1049l\033[?25h')
        if self._old_termios:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)
            except (termios.error, OSError):
                pass
            self._old_termios = None
        if self._old_sigwinch is not None:
            try:
                signal.signal(signal.SIGWINCH, self._old_sigwinch)
            except (OSError, ValueError):
                pass

    def _on_resize(self, signum, frame):
        size = shutil.get_terminal_size()
        self.cols, self.rows = size.columns, size.lines
        self._write('\033[2J')
        self._write(f'\033[{self._content_start};{self.rows}r')
        self._redraw()

    # ── Low-level output ────────────────────────────────────────

    def _write(self, data: str):
        with self._write_lock:
            sys.stdout.write(data)
            sys.stdout.flush()

    def _write_raw(self, data: str):
        """Write without lock — caller must hold _write_lock."""
        sys.stdout.write(data)

    # ── Header (rows 1-4) ──────────────────────────────────────

    def _draw_header(self):
        w = self.cols
        step = f"step {self.step_num}/{self.max_steps}" if self.step_num else ""

        # Row 1
        r1_text = f"─ OpenProver v{__version__}"
        if step:
            r1_text += f" ── {step}"
        r1_text += " "
        fill1 = max(w - len(r1_text) - 2, 0)
        self._write_raw('\033[1;1H\033[2K')
        self._write_raw(f'{BLUE}╭─{RESET} {BOLD}OpenProver{RESET} {DIM}v{__version__}{RESET}')
        if step:
            self._write_raw(f' {BLUE}──{RESET} {DIM}{step}{RESET}')
        self._write_raw(f' {BLUE}{"─" * fill1}╮{RESET}')

        # Row 2 — theorem
        name = (self.theorem_name or "").replace("\n", " ").replace("\r", "")
        max_name = max(w - 4, 10)
        if len(name) > max_name:
            display_name = name[:max_name - 3] + "..."
        else:
            display_name = name
        name_pad = max(w - 3 - len(display_name), 0)
        self._write_raw('\033[2;1H\033[2K')
        self._write_raw(f'{BLUE}│{RESET} {WHITE}{display_name}{RESET}')
        self._write_raw(f'{" " * name_pad}{BLUE}│{RESET}')

        # Row 3 — hints
        help_style = BOLD if self.view == "help" else DIM
        trace_style = BOLD if self.trace_visible else DIM
        wb_style = BOLD if self.view == "whiteboard" else DIM
        auto_style = BOLD if self.autonomous else DIM
        hints_styled = (f'{help_style}? help{RESET} {DIM}·{RESET} '
                        f'{trace_style}t trace{RESET} {DIM}·{RESET} '
                        f'{wb_style}w whiteboard{RESET} {DIM}·{RESET} '
                        f'{auto_style}a autonomous{RESET}')
        hints_len = len("? help · t trace · w whiteboard · a autonomous")
        pad = max(w - 2 - hints_len - 1, 0)
        self._write_raw('\033[3;1H\033[2K')
        self._write_raw(f'{BLUE}│{RESET}{" " * pad}{hints_styled} {BLUE}│{RESET}')

        # Row 4
        self._write_raw('\033[4;1H\033[2K')
        self._write_raw(f'{BLUE}╰{"─" * max(w - 2, 0)}╯{RESET}')

    def update_step(self, step_num: int, max_steps: int):
        self.step_num = step_num
        self.max_steps = max_steps
        with self._write_lock:
            self._write_raw('\033[s')
            self._draw_header()
            self._write_raw('\033[u')
            sys.stdout.flush()

    # ── Content area (scroll region) ───────────────────────────

    def _log(self, text: str, step_idx: int = -1):
        entry = _LogEntry(text, step_idx)
        self.log_lines.append(entry)
        if len(self.log_lines) > 200:
            self.log_lines = self.log_lines[-200:]
        if self.view == "main":
            self._write(f' {text}\n')

    def log(self, text: str, color: str = "", bold: bool = False, dim: bool = False):
        self._log(self._style(text, color, bold, dim))

    def show_proposal(self, plan: dict):
        # Separator between history and proposal
        sep = f'{DIM}{"─" * max(self.cols - 4, 20)}{RESET}'
        self._log(sep)
        self._log(f'{DIM}Next step:{RESET}')

        action = plan.get("action", "")
        summary = plan.get("summary", "")
        color = ACTION_STYLE.get(action, "")
        self._log(f'{color}▸{RESET} {BOLD}{action}{RESET} {DIM}—{RESET} {summary}')
        if plan.get("reasoning"):
            self._log(f'  {DIM}{plan["reasoning"]}{RESET}')

    def step_complete(self, step_num: int, max_steps: int,
                      action: str, summary: str, detail: str = ""):
        color = ACTION_STYLE.get(action, "")
        line = f'{color}■{RESET} {BOLD}{action}{RESET} {DIM}—{RESET} {summary}'
        # Save trace before clearing
        trace = "".join(self.trace_buf)
        self.trace_buf = []
        idx = len(self.step_entries)
        self._log(line, step_idx=idx)
        self.step_entries.append({
            "action": action, "summary": summary,
            "step_num": step_num, "detail": detail,
            "trace": trace,
        })
        self.update_step(step_num, max_steps)
        if self.view == "main":
            self._redraw()

    def update_step_detail(self, step_idx: int, detail: str):
        """Update the detail field of an existing step entry."""
        if 0 <= step_idx < len(self.step_entries):
            self.step_entries[step_idx]["detail"] = detail

    # ── Spinner ─────────────────────────────────────────────────

    def _start_spinner(self, label: str = "thinking"):
        self._spinner_label = label
        self._spinner_tick = 0
        self._spinner_time = time.monotonic()
        if self.view == "main":
            self._write(f'  {DIM}{SPINNER[0]} {label}{RESET}')

    def _update_spinner(self):
        now = time.monotonic()
        if now - self._spinner_time < 0.08:
            return
        self._spinner_time = now
        self._spinner_tick = (self._spinner_tick + 1) % len(SPINNER)
        if self.view == "main":
            ch = SPINNER[self._spinner_tick]
            with self._write_lock:
                # Don't overwrite trace text; only spin when no trace yet
                if not self._spinner_label or self.trace_buf:
                    return
                self._write_raw(f'\r\033[2K  {DIM}{ch} {self._spinner_label}{RESET}')
                sys.stdout.flush()

    def _stop_spinner(self):
        """Clear spinner from screen, reset label."""
        with self._write_lock:
            if self._spinner_label and self.view == "main":
                self._write_raw('\r\033[2K')
                sys.stdout.flush()
            self._spinner_label = ""

    # ── Streaming ───────────────────────────────────────────────

    def stream_start(self, label: str = "thinking"):
        self.trace_buf = []
        self.streaming = True
        self._start_spinner(label)

    def stream_text(self, text: str):
        self._check_keys()
        if not self.trace_buf:
            # First chunk — replace spinner with trace (or keep spinner if trace off)
            if self.trace_visible and self.view == "main":
                self._stop_spinner()
        self.trace_buf.append(text)
        if self.trace_visible and self.view == "main":
            self._write(f'{DIM}{text}{RESET}')

    def stream_end(self):
        self.streaming = False
        self._spinner_label = ""
        if self.trace_visible and self.view == "main":
            self._write('\n')
        elif self.view == "main":
            # Trace was off — clear the spinner that was still showing
            self._write('\r\033[2K')

    # ── Background thread (key reader + spinner) ────────────────

    def _bg_loop(self):
        fd = sys.stdin.fileno()
        while not self._bg_stop:
            try:
                if self._spinner_label and (self.streaming):
                    self._update_spinner()

                if not select.select([fd], [], [], 0.04)[0]:
                    continue
                data = os.read(fd, 32)
                if not data:
                    continue

                i = 0
                while i < len(data):
                    b = data[i]
                    if b == 0x1b:
                        if i + 2 < len(data) and data[i + 1] == 0x5b:
                            self._key_queue.put(chr(0x1b) + '[' + chr(data[i + 2]))
                            i += 3
                            continue
                        if self._can_handle_directly() and self.view != "main":
                            self.view = "main"
                            self._redraw()
                        else:
                            self._key_queue.put('\x1b')
                        i += 1
                        continue

                    ch = chr(b)
                    if self._can_handle_directly():
                        if ch in ('t', 'w', '?', 'a'):
                            self._process_key(ch)
                            i += 1
                            continue
                        if ch in ('\n', '\r') and self.view != "main":
                            self._process_key(ch)
                            i += 1
                            continue

                    self._key_queue.put(ch)
                    i += 1
            except (OSError, ValueError):
                break

    def _can_handle_directly(self) -> bool:
        return self.streaming and not self._confirming

    # ── Key handling ────────────────────────────────────────────

    def _check_keys(self):
        while True:
            try:
                ch = self._key_queue.get_nowait()
            except queue.Empty:
                break
            self._process_key(ch)

    def _process_key(self, ch: str):
        if ch == 't':
            self._toggle_trace()
        elif ch == 'w':
            self._toggle_view("whiteboard")
        elif ch == '?':
            self._toggle_view("help")
        elif ch == 'a':
            self.autonomous = not self.autonomous
            self._redraw()
        elif ch == '\x1b' and self.view != "main":
            self.view = "main"
            self._redraw()
        elif ch in ('\n', '\r') and self.view != "main":
            self.view = "main"
            self._redraw()
        elif self.autonomous and ch in ('q', 'p', 's'):
            self.pending_action = {
                'q': 'quit', 'p': 'pause',
                's': 'summarize',
            }[ch]

    def get_pending_action(self) -> str | None:
        self._check_keys()
        action = self.pending_action
        self.pending_action = None
        return action

    def interrupt(self):
        """Cancel any pending confirmation by injecting ctrl+c into key queue."""
        self._key_queue.put('\x03')

    # ── Confirmation UI ────────────────────────────────────────

    def get_confirmation(self) -> str:
        self._confirming = True
        self._confirm_selected = 0
        self._confirm_buf = []
        self._nav_step = -1
        self._redraw()
        self._write('\033[?25h')

        try:
            while True:
                try:
                    ch = self._key_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if ch == '\x1b':
                    if self.view != "main":
                        self.view = "main"
                    elif self._nav_step >= 0:
                        self._nav_step = -1
                    else:
                        self._confirm_buf.clear()
                    self._redraw()
                    continue

                if len(ch) == 3 and ch[:2] == '\x1b[':
                    if ch[2] == 'A':
                        self._nav_up()
                    elif ch[2] == 'B':
                        self._nav_down()
                    self._redraw()
                    continue

                if ch in ('\n', '\r'):
                    if self.view != "main":
                        self.view = "main"
                        self._redraw()
                        continue
                    if self._nav_step >= 0:
                        entry = self.step_entries[self._nav_step]
                        self._step_detail_title = (
                            f"Step {entry['step_num']}: {entry['action']}"
                            f" — {entry['summary']}"
                        )
                        # Build detail: trace (if visible) + action-specific info
                        parts = []
                        trace = entry.get("trace", "")
                        if trace and self.trace_visible:
                            parts.append(trace.rstrip())
                            parts.append("")
                        detail = entry.get("detail", "")
                        if detail:
                            parts.append(detail)
                        self._step_detail_text = "\n".join(parts) if parts else "(no detail)"
                        self.view = "step_detail"
                        self._redraw()
                        continue
                    if self._confirm_selected == 0:
                        return ""
                    return "".join(self._confirm_buf)

                if ch in ('\x7f', '\x08'):
                    if self._confirm_selected == 1 and self._confirm_buf:
                        self._confirm_buf.pop()
                        self._redraw()
                    continue

                if ch == '\x03':
                    raise KeyboardInterrupt
                if ch == '\x04':
                    raise EOFError

                if ch == '\t':
                    if self._nav_step >= 0:
                        self._nav_step = -1
                    else:
                        self._confirm_selected = 1 - self._confirm_selected
                    self._redraw()
                    continue

                can_toggle = (self._confirm_selected == 0 or not self._confirm_buf)
                if can_toggle and ch in ('t', 'w', '?'):
                    self._process_key(ch)
                    continue

                if (self._nav_step == -1 and self._confirm_selected == 0
                        and ch in ('s', 'p', 'q')):
                    return ch

                if (ch == 'a' and self._nav_step == -1
                        and self._confirm_selected == 0):
                    self.autonomous = True
                    self._redraw()
                    return "a"

                if ch.isprintable():
                    if self._nav_step >= 0:
                        self._nav_step = -1
                    if self._confirm_selected == 0:
                        self._confirm_selected = 1
                    self._confirm_buf.append(ch)
                    self._redraw()

        finally:
            self._confirming = False
            self._nav_step = -1
            self._redraw()
            self._write('\033[?25l')

    def _nav_up(self):
        if self._nav_step == -1:
            if self.step_entries:
                self._nav_step = len(self.step_entries) - 1
        elif self._nav_step > 0:
            self._nav_step -= 1

    def _nav_down(self):
        if self._nav_step >= 0:
            if self._nav_step < len(self.step_entries) - 1:
                self._nav_step += 1
            else:
                self._nav_step = -1

    def _draw_confirmation(self):
        fb = "".join(self._confirm_buf)
        self._write_raw('\n')
        if self._nav_step >= 0:
            self._write_raw(f' {DIM}○ accept{RESET}\n')
            self._write_raw(f' {DIM}○ give feedback{RESET}')
        elif self._confirm_selected == 0:
            self._write_raw(f' {GREEN}●{RESET} {BOLD}accept{RESET}\n')
            self._write_raw(f' {DIM}○ give feedback{RESET}')
        else:
            self._write_raw(f' {DIM}○ accept{RESET}\n')
            self._write_raw(f' {GREEN}●{RESET} {fb}')

    # ── View toggles ────────────────────────────────────────────

    def _toggle_trace(self):
        self.trace_visible = not self.trace_visible
        if self.view == "main":
            self._redraw()

    def _toggle_view(self, target: str):
        self.view = "main" if self.view == target else target
        self._redraw()

    # ── Redraw ──────────────────────────────────────────────────

    def _redraw(self):
        with self._write_lock:
            self._write_raw('\033[?25l')
            self._draw_header()
            cs = self._content_start
            for row in range(cs, self.rows + 1):
                self._write_raw(f'\033[{row};1H\033[2K')
            self._write_raw(f'\033[{cs};1H')

            if self.view == "main":
                for entry in self.log_lines:
                    is_step = entry.step_idx >= 0
                    if is_step and self._confirming and entry.step_idx == self._nav_step:
                        self._write_raw(f' {GREEN}▎{RESET}{entry.text}\n')
                    else:
                        self._write_raw(f' {entry.text}\n')
                # Spinner or trace
                if self._spinner_label and (self.streaming):
                    if not self.trace_visible or not self.trace_buf:
                        # Show spinner (trace off, or no text yet)
                        ch = SPINNER[self._spinner_tick]
                        self._write_raw(f'  {DIM}{ch} {self._spinner_label}{RESET}')
                    else:
                        # Trace on and we have text — show trace
                        for chunk in self.trace_buf:
                            self._write_raw(f'{DIM}{chunk}{RESET}')
                elif self.trace_buf:
                    if self.trace_visible:
                        for chunk in self.trace_buf:
                            self._write_raw(f'{DIM}{chunk}{RESET}')
                        if not self.streaming:
                            self._write_raw('\n')
                if self._confirming:
                    self._draw_confirmation()
                    self._write_raw('\033[?25h')
            elif self.view == "whiteboard":
                self._write_raw(f'  {BOLD}Whiteboard{RESET} {DIM}(esc to return){RESET}\n')
                self._write_raw(f'  {DIM}{"─" * 40}{RESET}\n')
                for wline in self.whiteboard.splitlines():
                    self._write_raw(f'  {wline}\n')
            elif self.view == "help":
                self._write_raw(HELP_TEXT)
            elif self.view == "step_detail":
                self._write_raw(f'  {BOLD}{self._step_detail_title}{RESET}')
                self._write_raw(f' {DIM}(esc to return){RESET}\n')
                self._write_raw(f'  {DIM}{"─" * 40}{RESET}\n')
                for dline in self._step_detail_text.splitlines():
                    self._write_raw(f'  {dline}\n')

            sys.stdout.flush()

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _style(text: str, color: str = "", bold: bool = False,
               dim: bool = False) -> str:
        prefix = ""
        if color:
            prefix += COLOR_MAP.get(color, "")
        if bold:
            prefix += BOLD
        if dim:
            prefix += DIM
        return f'{prefix}{text}{RESET}' if prefix else text
