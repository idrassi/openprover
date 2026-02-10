"""Terminal UI for OpenProver — ANSI scroll regions, fixed header, inline input."""

import queue
import select
import shutil
import signal
import sys
import termios
import threading
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

  {DIM}Instant keys:{RESET}
    t           toggle reasoning trace
    w           toggle whiteboard view
    ?           this help
    enter       dismiss overlay

  {DIM}When confirming a plan:{RESET}
    {DIM}tab/arrows{RESET}  switch accept / feedback
    enter       confirm selection
    s           summarize progress
    a           switch to autonomous mode
    p           pause (resume with --run-dir)
    r           restart proof search
    q           quit

  {DIM}In autonomous mode all keys are instant.{RESET}
  {DIM}Press ? or enter to dismiss.{RESET}
"""

COLOR_MAP = {
    "red": RED, "green": GREEN, "blue": BLUE,
    "yellow": YELLOW, "magenta": MAGENTA, "cyan": CYAN,
}


class TUI:
    def __init__(self):
        self.rows = 0
        self.cols = 0
        self.log_lines: list[str] = []
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
        self._key_thread: threading.Thread | None = None
        self._key_stop = False
        self._thinking = False
        # Confirmation state
        self._confirming = False
        self._confirm_selected = 0  # 0=accept, 1=feedback
        self._confirm_buf: list[str] = []

    def setup(self, theorem_name: str, work_dir: str,
              step_num: int = 0, max_steps: int = 50):
        self.theorem_name = theorem_name
        self.step_num = step_num
        self.max_steps = max_steps
        self.rows, self.cols = shutil.get_terminal_size()

        # Save terminal state and enter cbreak
        try:
            self._old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except (termios.error, OSError):
            self._old_termios = None

        # Alternate screen + clear
        self._write('\033[?1049h\033[2J')
        self._draw_header()
        # Scroll region: row 4 to bottom (header is rows 1-3)
        self._write(f'\033[4;{self.rows}r')
        # Cursor into scroll region, hide cursor by default
        self._write('\033[4;1H\033[?25l')
        self._active = True

        # Resize handler
        self._old_sigwinch = signal.getsignal(signal.SIGWINCH)
        signal.signal(signal.SIGWINCH, self._on_resize)

        # Start background key reader
        self._key_stop = False
        self._key_thread = threading.Thread(target=self._key_reader, daemon=True)
        self._key_thread.start()

    def cleanup(self):
        if not self._active:
            return
        self._active = False
        # Stop background key reader
        self._key_stop = True
        if self._key_thread:
            self._key_thread.join(timeout=0.2)
            self._key_thread = None
        # Reset scroll region, exit alternate screen, show cursor
        self._write('\033[r\033[?1049l\033[?25h')
        # Restore terminal
        if self._old_termios:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)
            except (termios.error, OSError):
                pass
            self._old_termios = None
        # Restore old SIGWINCH handler
        if self._old_sigwinch is not None:
            try:
                signal.signal(signal.SIGWINCH, self._old_sigwinch)
            except (OSError, ValueError):
                pass

    def _on_resize(self, signum, frame):
        self.rows, self.cols = shutil.get_terminal_size()
        self._write('\033[2J')
        self._draw_header()
        self._write(f'\033[4;{self.rows}r')
        self._redraw()

    # ── Low-level output ────────────────────────────────────────

    @staticmethod
    def _write(data: str):
        sys.stdout.write(data)
        sys.stdout.flush()

    # ── Header (rows 1-3) ─────────────────────────────────────

    def _draw_header(self):
        w = self.cols
        name = self.theorem_name[:40] if self.theorem_name else ""
        step = f"step {self.step_num}/{self.max_steps}" if self.step_num else ""

        # Measure visible length for row 1 fill
        title_len = 2 + len(f"OpenProver v{__version__}")  # "─ OpenProver v0.1.0"
        if name:
            title_len += 4 + len(name)  # "  ──  name" but ── is 2 chars visible
        if step:
            title_len += 4 + len(step)
        title_len += 1  # trailing space
        fill = max(w - title_len - 2, 0)  # -2 for ╭ and ╮

        # Row 1: ╭─ OpenProver v0.1.0 ── name ── step ────────╮
        self._write('\033[1;1H\033[2K')
        self._write(f'{BLUE}╭─{RESET} {BOLD}OpenProver{RESET} {DIM}v{__version__}{RESET}')
        if name:
            self._write(f' {BLUE}──{RESET} {WHITE}{name}{RESET}')
        if step:
            self._write(f' {BLUE}──{RESET} {DIM}{step}{RESET}')
        self._write(f' {BLUE}{"─" * fill}╮{RESET}')

        # Row 2: │                  ? help · t trace · w whiteboard │
        hints = "? help · t trace · w whiteboard"
        inner = w - 2  # space between │ and │
        pad = max(inner - len(hints) - 1, 0)
        self._write('\033[2;1H\033[2K')
        self._write(f'{BLUE}│{RESET}')
        self._write(f'{" " * pad}{DIM}{hints}{RESET} ')
        self._write(f'{BLUE}│{RESET}')

        # Row 3: ╰──────────────────────────────────────────────╯
        self._write('\033[3;1H\033[2K')
        self._write(f'{BLUE}╰{"─" * max(w - 2, 0)}╯{RESET}')

    def update_step(self, step_num: int, max_steps: int):
        self.step_num = step_num
        self.max_steps = max_steps
        self._write('\033[s')
        self._draw_header()
        self._write('\033[u')

    # ── Content area (scroll region) ───────────────────────────

    def log(self, text: str, color: str = "", bold: bool = False, dim: bool = False):
        styled = self._style(text, color, bold, dim)
        self.log_lines.append(styled)
        if len(self.log_lines) > 200:
            self.log_lines = self.log_lines[-200:]
        if self.view == "main":
            self._write(f' {styled}\n')

    def show_proposal(self, plan: dict):
        action = plan.get("action", "")
        summary = plan.get("summary", "")
        color = ACTION_STYLE.get(action, "")
        line = f'{color}▸{RESET} {BOLD}{action}{RESET} {DIM}—{RESET} {summary}'
        self.log_lines.append(line)
        if self.view == "main":
            self._write(f' {line}\n')
        if plan.get("reasoning"):
            r_line = f'  {DIM}{plan["reasoning"]}{RESET}'
            self.log_lines.append(r_line)
            if self.view == "main":
                self._write(f' {r_line}\n')

    def step_complete(self, step_num: int, max_steps: int,
                      action: str, summary: str):
        color = ACTION_STYLE.get(action, "")
        line = f'{color}■{RESET} {BOLD}{action}{RESET} {DIM}—{RESET} {summary}'
        self.trace_buf = []
        self.log_lines.append(line)
        self.update_step(step_num, max_steps)
        if self.view == "main":
            self._redraw()

    # ── Streaming ───────────────────────────────────────────────

    def stream_start(self):
        self.trace_buf = []
        self.streaming = True

    def stream_text(self, text: str):
        # Check keys BEFORE appending to trace_buf to prevent
        # duplication if _check_keys triggers a _redraw
        self._check_keys()
        self.trace_buf.append(text)
        if self.trace_visible and self.view == "main":
            self._write(f'{DIM}{text}{RESET}')

    def stream_end(self):
        self.streaming = False
        if self.trace_visible and self.view == "main":
            self._write('\n')

    # ── Thinking (non-streaming calls like plan) ────────────────

    def thinking(self):
        self.streaming = True
        self._thinking = True
        self.trace_buf = []
        if self.view == "main":
            self._write(f'  {DIM}thinking...{RESET}')

    def thinking_done(self):
        self._thinking = False
        self.streaming = False
        if self.view == "main":
            self._write('\r\033[2K')
        # Process any keys that were queued during thinking
        self._check_keys()

    # ── Background key reader ──────────────────────────────────

    def _key_reader(self):
        """Background thread: read stdin chars, queue or process them."""
        while not self._key_stop:
            try:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if not ch:
                        continue
                    # During thinking, main thread is blocked — safe to
                    # handle view toggles directly from this thread
                    if self._thinking and ch in ('t', 'w', '?'):
                        self._process_key(ch)
                    elif (self._thinking and ch in ('\n', '\r')
                          and self.view != "main"):
                        self._process_key(ch)
                    else:
                        self._key_queue.put(ch)
            except (OSError, ValueError):
                break

    # ── Key handling ────────────────────────────────────────────

    def _check_keys(self):
        """Drain the key queue and process each key."""
        while True:
            try:
                ch = self._key_queue.get_nowait()
            except queue.Empty:
                break
            self._process_key(ch)

    def _process_key(self, ch: str):
        """Handle a single key press (view toggles, autonomous commands)."""
        if ch == 't':
            self._toggle_trace()
        elif ch == 'w':
            self._toggle_view("whiteboard")
        elif ch == '?':
            self._toggle_view("help")
        elif ch in ('\n', '\r') and self.view != "main":
            self.view = "main"
            self._redraw()
        elif self.autonomous and ch in ('q', 'p', 'r', 'i', 's'):
            self.pending_action = {
                'q': 'quit', 'p': 'pause', 'r': 'restart',
                'i': 'interactive', 's': 'summarize',
            }[ch]

    def get_pending_action(self) -> str | None:
        self._check_keys()
        action = self.pending_action
        self.pending_action = None
        return action

    # ── Confirmation UI ────────────────────────────────────────

    def get_confirmation(self) -> str:
        """Two-option confirmation. Returns "" for accept, feedback text,
        or single command char (s/a/p/r/q)."""
        self._confirming = True
        self._confirm_selected = 0
        self._confirm_buf = []

        # Draw initial confirmation
        self._write('\033[s')  # save cursor (start of confirmation)
        self._draw_confirmation()
        self._write('\033[?25h')  # show cursor

        try:
            while True:
                try:
                    ch = self._key_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if ch == '\x1b':
                    # Escape sequence — check for arrow keys
                    try:
                        ch2 = self._key_queue.get(timeout=0.05)
                    except queue.Empty:
                        # Plain escape — clear feedback buf
                        self._confirm_buf.clear()
                        self._update_confirmation()
                        continue
                    if ch2 == '[':
                        try:
                            ch3 = self._key_queue.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        if ch3 == 'A':  # Up arrow
                            self._confirm_selected = 0
                        elif ch3 == 'B':  # Down arrow
                            self._confirm_selected = 1
                        self._update_confirmation()
                    # Drain any remaining escape bytes
                    while True:
                        try:
                            self._key_queue.get_nowait()
                        except queue.Empty:
                            break

                elif ch in ('\n', '\r'):
                    if self.view != "main":
                        self.view = "main"
                        self._redraw()
                        continue
                    if self._confirm_selected == 0:
                        return ""
                    else:
                        return "".join(self._confirm_buf)

                elif ch in ('\x7f', '\x08'):  # backspace
                    if self._confirm_selected == 1 and self._confirm_buf:
                        self._confirm_buf.pop()
                        self._update_confirmation()

                elif ch == '\x03':
                    raise KeyboardInterrupt

                elif ch == '\x04':
                    raise EOFError

                elif ch == '\t':
                    # Tab switches between accept/feedback regardless of state
                    self._confirm_selected = 1 - self._confirm_selected
                    self._update_confirmation()
                    continue

                # View toggles (on accept line, or feedback with empty buf)
                elif (ch == 't'
                      and (self._confirm_selected == 0
                           or not self._confirm_buf)):
                    self._toggle_trace()
                    continue
                elif (ch == 'w'
                      and (self._confirm_selected == 0
                           or not self._confirm_buf)):
                    self._toggle_view("whiteboard")
                    continue
                elif (ch == '?'
                      and (self._confirm_selected == 0
                           or not self._confirm_buf)):
                    self._toggle_view("help")
                    continue

                # Command hotkeys on accept line
                elif (self._confirm_selected == 0
                      and ch in ('s', 'a', 'p', 'r', 'q')):
                    return ch

                elif ch.isprintable():
                    if self._confirm_selected == 0:
                        # Auto-switch to feedback
                        self._confirm_selected = 1
                        self._confirm_buf.append(ch)
                    else:
                        self._confirm_buf.append(ch)
                    self._update_confirmation()

        finally:
            self._confirming = False
            self._write('\033[?25l')

    def _draw_confirmation(self):
        """Write the two-option confirmation at the current cursor."""
        fb = "".join(self._confirm_buf)
        self._write('\n')
        if self._confirm_selected == 0:
            self._write(f' {GREEN}●{RESET} {BOLD}accept{RESET}\n')
            self._write(f' {DIM}○ give feedback{RESET}')
        else:
            self._write(f' {DIM}○ accept{RESET}\n')
            self._write(f' {GREEN}●{RESET} {fb}')

    def _update_confirmation(self):
        """Efficient redraw of just the confirmation area."""
        self._write('\033[u')  # restore to saved position
        self._write('\033[J')  # clear from cursor to end of scroll region
        self._draw_confirmation()

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
        # Hide cursor during redraw
        self._write('\033[?25l')
        # Clear scroll region
        for row in range(4, self.rows + 1):
            self._write(f'\033[{row};1H\033[2K')
        self._write('\033[4;1H')

        if self.view == "main":
            for line in self.log_lines:
                self._write(f' {line}\n')
            if self._thinking:
                self._write(f'  {DIM}thinking...{RESET}')
            elif self.trace_buf:
                if self.trace_visible:
                    for chunk in self.trace_buf:
                        self._write(f'{DIM}{chunk}{RESET}')
                    # No trailing newline during streaming so cursor stays
                    # at end of trace for seamless continuation
                    if not self.streaming:
                        self._write('\n')
                elif self.streaming:
                    self._write(f'  {DIM}thinking...{RESET}')
            if self._confirming:
                self._write('\033[s')  # save cursor for confirmation updates
                self._draw_confirmation()
                self._write('\033[?25h')  # show cursor during confirmation
        elif self.view == "whiteboard":
            self._write(f'  {BOLD}Whiteboard{RESET} {DIM}(press w or enter to return){RESET}\n')
            self._write(f'  {DIM}{"─" * 40}{RESET}\n')
            for wline in self.whiteboard.splitlines():
                self._write(f'  {wline}\n')
        elif self.view == "help":
            self._write(HELP_TEXT)

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
