"""Playground: investigate and fix ctrl+c behaviour in openprover.

Root cause (works for parallelism=1 too)
-----------------------------------------
After soft-interrupt, Phase 2 runs with the same stream_callback as Phase 1.
Claude still does extended thinking in Phase 2, and that thinking is streamed
to the TUI just like Phase 1 was — so from the user's perspective the
"reasoning continues" after ctrl+c.  Fix: Phase 2 uses output_only=True on
_stream_cb(), which filters out thinking tokens so only output text is shown.

Additional bug – Multi-worker clear_soft_interrupt race (parallelism > 1)
--------------------------------------------------------------------------
All workers share one _soft_interrupted Event.  Worker-A finishes Phase 1,
calls clear_soft_interrupt(), then Worker-B's race-condition fallback check
sees the flag already cleared → raises RuntimeError instead of starting
Phase 2.  Fix: also treat proc.returncode < 0 (SIGKILL) with no result as
soft_interrupted, regardless of the shared flag state.

Also fixed – Non-atomic debounce in request_interrupt()
--------------------------------------------------------
The TUI bg-thread and the SIGINT signal handler can both call
request_interrupt().  A threading.Lock makes the check+increment atomic.
CPython's GIL makes the race rare in practice, but the lock is correct.

Run directly:  python tests/test_interrupt_race.py
"""

import os
import signal
import subprocess
import threading
import time


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(label: str, ok: bool):
    print(f"  [{PASS if ok else FAIL}] {label}")
    if not ok:
        raise AssertionError(label)


# ─────────────────────────────────────────────────────────────────────────────
# Bug 1 demo: multi-worker clear_soft_interrupt race
# ─────────────────────────────────────────────────────────────────────────────

class SoftInterruptEvent:
    """Minimal replica of the _soft_interrupted flag in llm/claude.py."""
    def __init__(self):
        self._ev = threading.Event()

    def set(self): self._ev.set()
    def is_set(self): return self._ev.is_set()
    def clear(self): self._ev.clear()


def simulate_worker(worker_id: int, flag: SoftInterruptEvent,
                    results: list, delay_before_phase2: float = 0.0):
    """
    Simulate _call_streaming for one worker.

    Phase 1: 'readline()' blocks until the process is killed (flag.set() + kill).
    After kill, readline returns ''.
    Race-condition fallback: check if flag is still set.

    Phase 2 (if detected): call clear_soft_interrupt() then 'run phase 2'.
    """
    # Simulate blocking in readline()
    soft_interrupted_local = False
    result_data = None  # no result received before kill

    # ... readline blocks until flag is set (simulated by event)
    flag._ev.wait()     # unblock when soft_interrupt() fires
    # readline() returns "" (EOF because process was killed)

    # ── current (buggy) race-condition check ─────────────────────────────────
    # Another worker may have already cleared the flag here.
    time.sleep(delay_before_phase2)  # simulate scheduling delay
    if not soft_interrupted_local and flag.is_set():
        soft_interrupted_local = True

    if soft_interrupted_local:
        # Phase 2: clear flag first, then do work
        flag.clear()
        results[worker_id] = "phase2"
    elif result_data is None:
        results[worker_id] = "RuntimeError"
    else:
        results[worker_id] = "ok"


def demo_multi_worker_race():
    print("\n── Bug 1: multi-worker clear_soft_interrupt race ────────────────")

    N_TRIALS = 500
    race_count = 0

    for _ in range(N_TRIALS):
        flag = SoftInterruptEvent()
        results = [None, None]

        # Worker 0 is slightly faster than Worker 1 (models scheduling jitter)
        t0 = threading.Thread(target=simulate_worker,
                              args=(0, flag, results, 0.0))
        t1 = threading.Thread(target=simulate_worker,
                              args=(1, flag, results, 0.000_05))  # 50µs slower

        t0.start(); t1.start()
        time.sleep(0.001)  # let workers reach readline() block

        flag.set()  # simulate soft_interrupt() firing

        t0.join(); t1.join()

        if results[1] == "RuntimeError":
            race_count += 1

    pct = race_count / N_TRIALS * 100
    print(f"  {race_count}/{N_TRIALS} trials: Worker-1 got RuntimeError "
          f"because Worker-0 cleared the flag first ({pct:.1f}%)")
    check("race is observable (> 0 occurrences)", race_count > 0)


def simulate_worker_fixed(worker_id: int, flag: SoftInterruptEvent,
                          results: list, proc_returncode: int,
                          delay_before_phase2: float = 0.0):
    """Fixed version: also check proc.returncode < 0 in the fallback."""
    soft_interrupted_local = False
    result_data = None

    flag._ev.wait()
    time.sleep(delay_before_phase2)

    # ── fixed race-condition check ────────────────────────────────────────────
    if not soft_interrupted_local:
        if flag.is_set():
            soft_interrupted_local = True
        elif result_data is None and proc_returncode < 0:
            # Process was killed by signal (SIGKILL = -9) with no result.
            # _soft_interrupted may have been cleared by a sibling worker, but
            # the negative returncode tells us a kill happened on our behalf.
            soft_interrupted_local = True

    if soft_interrupted_local:
        flag.clear()   # safe to call even if already cleared
        results[worker_id] = "phase2"
    elif result_data is None:
        results[worker_id] = "RuntimeError"
    else:
        results[worker_id] = "ok"


def demo_multi_worker_race_fixed():
    print("\n── Bug 1 (fixed): returncode check eliminates false RuntimeError ─")

    N_TRIALS = 500
    race_count = 0

    for _ in range(N_TRIALS):
        flag = SoftInterruptEvent()
        results = [None, None]

        t0 = threading.Thread(target=simulate_worker_fixed,
                              args=(0, flag, results, -9, 0.0))
        t1 = threading.Thread(target=simulate_worker_fixed,
                              args=(1, flag, results, -9, 0.000_05))

        t0.start(); t1.start()
        time.sleep(0.001)
        flag.set()
        t0.join(); t1.join()

        if results[1] == "RuntimeError":
            race_count += 1

    pct = race_count / N_TRIALS * 100
    print(f"  {race_count}/{N_TRIALS} trials: Worker-1 got RuntimeError ({pct:.1f}%)")
    check("fixed: no RuntimeError from race", race_count == 0)


# ─────────────────────────────────────────────────────────────────────────────
# Bug 2 demo: non-atomic debounce
# ─────────────────────────────────────────────────────────────────────────────

class BrokenRequestInterrupt:
    def __init__(self):
        self._workers_active = True
        self._interrupt_count = 0
        self._last_interrupt_time = 0.0
        self.actions: list[str] = []

    def request_interrupt(self):
        now = time.time()
        if now - self._last_interrupt_time < 0.1:
            return
        # ← NOT atomic: another thread can pass the check before we write here
        self._last_interrupt_time = now
        self._interrupt_count += 1

        if self._workers_active and self._interrupt_count == 1:
            self.actions.append("soft")
            return
        if self._interrupt_count >= 3:
            self.actions.append("shutdown")
            return
        self.actions.append("hard")


class FixedRequestInterrupt:
    def __init__(self):
        self._workers_active = True
        self._interrupt_count = 0
        self._last_interrupt_time = 0.0
        self._interrupt_lock = threading.Lock()
        self.actions: list[str] = []

    def request_interrupt(self):
        now = time.time()
        with self._interrupt_lock:
            if now - self._last_interrupt_time < 0.1:
                return
            self._last_interrupt_time = now
            self._interrupt_count += 1
            count = self._interrupt_count
            workers_active = self._workers_active

        if workers_active and count == 1:
            self.actions.append("soft")
            return
        if count >= 3:
            self.actions.append("shutdown")
            return
        self.actions.append("hard")


def demo_debounce_race(N: int = 5000):
    print(f"\n── Bug 2: debounce race ({N} simultaneous-pair trials) ──────────")

    broken_wrong = 0
    fixed_wrong = 0
    broken = BrokenRequestInterrupt()
    fixed = FixedRequestInterrupt()

    for _ in range(N):
        for obj in (broken, fixed):
            obj._workers_active = True
            obj._interrupt_count = 0
            obj._last_interrupt_time = 0.0
            obj.actions.clear()

        barrier = threading.Barrier(2)

        def call(obj):
            barrier.wait()
            obj.request_interrupt()

        for obj in (broken, fixed):
            t1 = threading.Thread(target=call, args=(obj,))
            t2 = threading.Thread(target=call, args=(obj,))
            t1.start(); t2.start()
            t1.join(); t2.join()

        if tuple(sorted(broken.actions)) != ('soft',):
            broken_wrong += 1
        if tuple(sorted(fixed.actions)) != ('soft',):
            fixed_wrong += 1

    print(f"  BrokenRequestInterrupt: {broken_wrong}/{N} wrong outcomes "
          f"({broken_wrong/N*100:.1f}%)")
    print(f"  FixedRequestInterrupt:  {fixed_wrong}/{N} wrong outcomes "
          f"({fixed_wrong/N*100:.1f}%)")
    # Note: CPython GIL makes the race hard to trigger; the lock is still the
    # correct fix for correctness.
    check("fixed version has zero wrong outcomes", fixed_wrong == 0)


# ─────────────────────────────────────────────────────────────────────────────
# Verify: killing a subprocess unblocks readline()
# ─────────────────────────────────────────────────────────────────────────────

def demo_kill_unblocks_readline():
    print("\n── Verify: os.killpg unblocks readline() ────────────────────────")

    proc = subprocess.Popen(
        ["python3", "-c",
         "import time, sys\n"
         "for i in range(200):\n"
         "    print(f'line {i}', flush=True)\n"
         "    time.sleep(0.02)\n"],
        stdout=subprocess.PIPE, text=True, bufsize=1,
        start_new_session=True,
    )

    lines_read: list[str] = []
    unblocked = threading.Event()

    def read_loop():
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            lines_read.append(line.strip())
        unblocked.set()

    reader = threading.Thread(target=read_loop, daemon=True)
    reader.start()
    time.sleep(0.15)   # let it produce ~7 lines

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        proc.kill()
    proc.wait()

    unblocked.wait(timeout=2.0)
    check("readline() unblocked within 2s of SIGKILL", unblocked.is_set())
    check("partial output captured (not all 200 lines)", 0 < len(lines_read) < 200)
    check("process returncode is negative (killed by signal)", proc.returncode < 0)
    print(f"  captured {len(lines_read)} lines, returncode={proc.returncode}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    demo_kill_unblocks_readline()
    demo_multi_worker_race()
    demo_multi_worker_race_fixed()
    demo_debounce_race()
    print("\n── All tests passed ─────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
