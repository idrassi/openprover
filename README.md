# OpenProver

Theorem prover powered by language models. Uses Claude CLI as a reasoning engine to explore proof strategies, verify arguments, and search mathematical literature.

## How it works

You give it a theorem statement (a `.md` file). It maintains a **whiteboard**, a terse scratchpad like a real mathematician's blackboard, and iterates through proving steps: exploring avenues, extracting lemmas, verifying arguments, and searching literature. Each step is planned, optionally reviewed by you, then executed.

Two modes:
- **Interactive** (default): you see each proposed step and can accept, give feedback, or redirect
- **Autonomous** (`--autonomous`): runs hands-off until it finds a proof or exhausts its step budget

## Requirements

- Python 3.10+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude` command available on PATH)

## Install

```bash
git clone https://github.com/yourusername/openprover.git
cd openprover
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Interactive mode
openprover examples/sqrt2_irrational.md

# Autonomous with Opus, 100 step budget
openprover examples/erdos_838.md --model opus --max-steps 100 --autonomous

# Resume an interrupted run
openprover --run-dir runs/sqrt2-irrational-20260217-143012

# Offline mode (no web searches)
openprover examples/cauchy_schwarz.md --isolation
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `sonnet` | Model to use (`sonnet` or `opus`) |
| `--max-steps` | `50` | Step budget |
| `--autonomous` | off | Run without human confirmation |
| `--run-dir` | | Resume from an existing run directory |
| `--isolation` | off | Disable literature search / web access |
| `--verbose` | off | Show full LLM responses |

### TUI controls

| Key | Action |
|-----|--------|
| `t` | Toggle reasoning trace |
| `w` | Toggle whiteboard view |
| `a` | Toggle autonomous mode |
| `?` | Help overlay |
| `s` | Summarize progress |
| `p` | Pause (resume later with `--run-dir`) |
| `r` | Restart proof search |
| `q` | Quit |

When confirming a step: Tab switches between accept/feedback, up/down browses step history, Enter on a step shows its detail.

## Output

Each run creates a directory under `runs/`:

- `THEOREM.md` - original theorem statement
- `WHITEBOARD.md` - current whiteboard state
- `PROOF.md` - final proof (if found)
- `DISCUSSION.md` - post-session analysis
- `lemmas/` - extracted lemmas with proofs
- `steps/` - per-step input/output snapshots
- `archive/calls/` - raw LLM call logs

## Example theorems

The `examples/` directory has theorem statements at various difficulty levels:

- `sqrt2_irrational.md` - warm-up
- `infinite_primes.md` - classic
- `cauchy_schwarz.md` - standard inequality
- `e_irrational.md` - analysis
- `erdos_205.md`, `erdos_838.md` - research-level combinatorics
- `collatz.md` - famously open
