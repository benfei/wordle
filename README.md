# WORDLE AGENT

An agent that plays Wordle by maximizing the entropy of every guess. Implemented in Python using `numba` for performance.

## Examples
```bash
python play_wordle.py  # Play Wordle.
python play_wordle.py --hard-mode  # Play Wordle in hard mode.
python play_wordle.py --secret-word="vague"  # Self-play Wordle.
python play_wordle.py --evaluate --report-progress  # Evaluate.
```

