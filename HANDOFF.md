# Handoff: reproducing Table 1 (row 1) of the CEVD paper

## Goal (current)
Reproduce Table 1's first row from "On Sustainable Ride Pooling Through
Conditional Expected Value Decomposition" (ECAI 2023): Baseline vs. NeurADP vs.
CEVD "Requests Served" (mean ± std over 5 test days) at 500 vehicles, capacity 4,
pickup delay 90s, decision epoch 60s, plus CEVD's % improvement over NeurADP.
Paper's numbers for this row: Baseline 85423 ± 2190, NeurADP 90286 ± 2108, CEVD
98748.2 ± 2449 (9.37% ± 0.59 improvement).

**Full plan and rationale: `claude_plan.md`** (repo root) — this file covers the
environment/infra work; `claude_plan.md` covers the paper cross-check findings,
the day-index reconstruction, every code fix, and execution order in detail.

## Status as of now
- Environment (TF/Keras, CPLEX) issues from the original "just run
  main_scoring.py" goal: **all fixed** — see Problems 1-3 below, still accurate.
- Cross-checked the repo's default config against the paper and found it doesn't
  match: only 2 training days used (paper: 8), only `num_clusters=2` in training
  (paper: K=100 for 500 vehicles — this is CEVD's core neighbor-clustering
  parameter, not cosmetic), evaluation swept all 20 days including training days
  instead of the paper's 5 held-out test days, and no Baseline implementation was
  being exercised at all. Full detail in `claude_plan.md`.
- Fixed `main_plus.py`, `main_vanilla.py`, `main_scoring.py` and added new
  `main_baseline.py` — see "Reproduction fixes" below.
- Three background jobs currently running (see "Current runs" below): Baseline
  scoring, NeurADP training, CEVD training.

## Summary of all changes made this session

**System-level (outside git):**
- Installed Python 3.7.9 (Intel) via the official python.org installer to
  `/Library/Frameworks/Python.framework/Versions/3.7` (user ran this themselves
  with `sudo`, not the assistant).
- Created `.venv/` (repo root, arm64, Python 3.9) — the main environment, has
  TensorFlow 2.15, Keras 2.15, numpy 1.26.4, docplex, pandas, scikit-learn,
  joblib, torch.
- Created `.venv-cplex/` (repo root, x86_64/Rosetta, Python 3.7) — solves ILPs
  only, via `cplex_bridge_worker.py`; has the real CPLEX 20.1 bindings and
  docplex installed from the local CPLEX Studio install (not PyPI).
- Created `models/` directory (didn't exist).

**Modified, tracked files (`git diff --stat`: 6 files, +36/-18 lines):**
- `.gitignore` — added `.venv/` (alongside the pre-existing `cevd_env/` entry).
- `src/CentralAgent.py` — `_choose_actions_ILP`: `model.solve()` → `cplex_solve(model)`.
- `src/Environment.py` — `_get_rebalance_targets`: same swap, plus the import.
- `src/main_plus.py` — `num_clusters` 2→100; `TRAINING_DAYS` `[4,9]` →
  `[2,3,4,7,8,9,10,11]`.
- `src/main_vanilla.py` — same `TRAINING_DAYS` fix; removed a stray `break`
  that silently truncated training to 1 day; replaced single-day scoring with
  a 5-test-day loop + mean/std; fixed a checkpoint-filename suffix bug (hard
  coded `1` → `len(TRAINING_DAYS)`).
- `src/main_scoring.py` — same `TRAINING_DAYS` fix; `for day in range(1,21):` →
  `for day in [14,15,16,17,18]:`; checkpoint-filename suffix `2` →
  `len(TRAINING_DAYS)`; added mean/std accumulation; (from earlier in the
  session) uncommented the `load_weights(...)` call that was originally
  commented out.

**New, untracked files:**
- `HANDOFF.md` (this file), `claude_plan.md` (the approved reproduction plan).
- `src/cplex_bridge.py`, `src/cplex_bridge_worker.py` — the CPLEX split-process
  bridge (runs in `.venv`, shells out to `.venv-cplex`).
- `src/main_baseline.py` — new script for the paper's Baseline (greedy
  `ImmediateReward`, no training needed).
- Two `.h5` checkpoints in `models/` from the now-superseded first training
  pass (2 days, `num_clusters=2`) — see "Problem 4" below; will be superseded
  by new 8-day/`K=100` checkpoints once the current training runs finish.

**Not touched by the assistant:** `src/my_readme` (untracked, pre-existing,
unrelated to this work — left alone).

## Problem 1: TensorFlow 2.20 / Keras 3 vs. this TF1-style codebase

The global Python environment (`~/Library/Python/3.9/...`, system `python3`) had
TensorFlow 2.20 + Keras 3.10 installed. This codebase is written against TF1-style
APIs (`tf.compat.v1.Session`, `tf.compat.v1.disable_eager_execution()`,
`tensorflow.python.keras.*` internals). Keras 3 doesn't honor those legacy flags —
it runs eagerly regardless — which broke mask broadcasting inside the CEVD model
(`BroadcastTo` op failing on a `None` batch dim in `CEVD.py`'s `get_value`).

**Fix:** created an isolated venv at `.venv/` (repo root, arm64, Python 3.9, added
to `.gitignore`) with pinned versions that still bundle legacy Keras 2:
```
tensorflow==2.15.0
keras==2.15.0
numpy==1.26.4      # TF 2.15 needs numpy <2.0
docplex
pandas
scikit-learn
joblib
torch              # only used for torch.utils.tensorboard
```
With TF 2.15, all the original `tensorflow.python.keras` / `tensorflow.keras.backend`
imports work unmodified — `src/CEVD.py` and `src/main_scoring.py` were reverted to
their originally-committed state (`git checkout -- src/CEVD.py src/main_scoring.py`).
No source changes were needed for this part; it was purely an environment fix.

## Problem 2: CPLEX Community Edition's 1000-variable cap

`docplex`/`cplex` installed from PyPI is the free Community Edition, capped at
1000 variables/constraints. The vehicle-rebalancing LP in
`Environment._get_rebalance_targets` (`src/Environment.py`) routinely builds
~190K-variable problems (500 agents × up to 500 targets), which the Community
Edition refuses to solve (`CPLEX Error 1016`).

The user has a real, licensed **CPLEX 20.1** install at
`/Applications/CPLEX_Optimizer201`.

## Problem 3: CPLEX 20.1 vs. TensorFlow can't share one Python process here

CPLEX 20.1's Python bindings only support **Python 3.7 or 3.8**, compiled
**x86_64-only** (`_pycplex_platform.py` explicitly raises on Python 3.9+, and the
`.so` files are Intel-only — confirmed via `file`/`lipo`). Our TF venv is Python
3.9 on Apple Silicon (arm64). Running that x86_64 CPLEX under Rosetta 2 seemed
like the fix, **but Rosetta 2 does not support AVX instructions, and every
published TensorFlow wheel since ~2018 requires AVX** — so TensorFlow crashes
immediately under Rosetta regardless of TF version (confirmed directly: "The
TensorFlow library was compiled to use AVX instructions, but these aren't
available on your machine").

**Net result:** TensorFlow (needs native arm64) and CPLEX 20.1 (needs x86_64/
Rosetta + Python ≤3.8) cannot coexist in the same Python process on this machine.

### Fix: split-process bridge

- Installed Python 3.7.9 (Intel, via the official python.org installer — the last
  3.7 release, matches CPLEX's required Python version and its Intel-only build)
  to `/Library/Frameworks/Python.framework/Versions/3.7`. Required `sudo`, was run
  by the user directly (not by the assistant, since the sandboxed shell can't
  supply a password interactively).
- Built a second venv, `.venv-cplex/` (repo root, x86_64/Rosetta, Python 3.7,
  also gitignored), on top of that interpreter. Installed the real CPLEX bindings
  and docplex from the local Studio install (not from PyPI):
  ```
  pip install /Applications/CPLEX_Optimizer201/cplex/python/3.7/x86-64_osx
  pip install /Applications/CPLEX_Optimizer201/python/docplex
  ```
  Verified this has no size limit (solved a problem the Community Edition
  rejected).
- Added two new files:
  - `src/cplex_bridge_worker.py` — tiny script that runs **inside `.venv-cplex`**
    (invoked as a subprocess, never imported directly). Takes an LP file path and
    an output path, solves with the real `cplex` package, writes a JSON result
    (`status_string`, `objective`, `values` keyed by variable name).
  - `src/cplex_bridge.py` — runs **inside the main TF venv (`.venv`)**. Its
    `solve(model)` function exports a docplex `Model` to a temp LP file
    (`model.export_as_lp` is pure Python, doesn't need a CPLEX engine to run),
    shells out to `.venv-cplex/bin/python3.7 cplex_bridge_worker.py`, reads back
    the JSON, and wraps it in a `BridgeSolution` object exposing `.get_value(var)`
    and `__bool__` — just enough of docplex's `SolveSolution` API for the two call
    sites below to keep working unmodified otherwise.
- Edited `src/CentralAgent.py` (`_choose_actions_ILP`, ~line 130) and
  `src/Environment.py` (`_get_rebalance_targets`, ~line 147): replaced
  `solution = model.solve()` with `solution = cplex_solve(model)` (imported from
  `cplex_bridge`). These are the only two places in `src/*.py` that called
  `.solve()` on a docplex model.
- Variable-name round-tripping was verified explicitly: docplex's LP export and
  `Var.get_name()` produce identical strings (including matrix-var names like
  `assignments_0_0`), so `BridgeSolution.get_value(var)` correctly looks values up
  by name without needing the original `Var` objects on the worker side.

With this in place, `main_scoring.py` ran past both the TF bug and the CPLEX
size-limit error and got into real per-timestep scoring
(`Reward for epoch: ...`, `Percentage Served: ...`).

## Problem 4 (earlier, now superseded): `main_scoring.py` never actually loaded weights

Originally found `value_function.model.load_weights(...)` commented out in
`main_scoring.py`, and a first training/scoring pass was done with `main_plus.py`
defaults (`TRAINING_DAYS = [4, 9]`, `num_clusters = 2`) to prove the pipeline
worked end to end. That run completed successfully (two `.h5` checkpoints were
saved), but cross-checking against the paper afterward showed that config isn't
representative of the paper's actual setup (see next section) — so those
checkpoints and that scoring run are now obsolete and were superseded by the
reproduction work below. Kept for history; not used going forward.

## Reproducing the paper's Table 1, row 1

Full rationale and day-index derivation are in `claude_plan.md`. Summary of what
changed in each file (all in `src/`):

- **`main_plus.py`** (trains CEVD's individual value network Qθ):
  `num_clusters` 2 → 100 (paper's K for 500 vehicles — this is CEVD's core
  neighbor-clustering granularity, not a cosmetic knob), `TRAINING_DAYS` `[4, 9]`
  → `[2, 3, 4, 7, 8, 9, 10, 11]` (paper: 8 training weekdays).
- **`main_vanilla.py`** (trains + scores NeurADP): same `TRAINING_DAYS` fix;
  removed a stray `break` right after `envt.num_days_trained += 1` that was
  silently truncating training to just the first day; replaced the single-day
  `for day in [args.plot]:` scoring block with a loop over the 5 test days
  `[14, 15, 16, 17, 18]`, accumulating each day's requests-served into a
  mean ± std printed at the end.
- **`main_scoring.py`** (applies CEVD's actual neighbor-value combination, using
  Qθ trained by `main_plus.py` plus the λ/ψ per-3-hour-interval constants
  already hardcoded here for exactly this config — `a_arr`/`alpha_arr` around
  line 75): same `TRAINING_DAYS` fix (must match `main_plus.py` — it's used to
  reconstruct the checkpoint filename to load), the hardcoded trailing `2` in
  the `train_file` format call → `len(TRAINING_DAYS)` (was reading the wrong,
  2-day-trained checkpoint's name), `for day in range(1,21):` → the same 5
  test days, plus the same mean ± std accumulation as `main_vanilla.py`.
- **New file `main_baseline.py`**: adapted from `main_vanilla.py`'s scoring
  path, but uses `CEVD.py`'s already-implemented `ImmediateReward` value
  function (a non-learned greedy matcher — `update`/`remember` are no-ops, so
  no training phase at all) as the paper's Baseline stand-in. Same 5-test-day
  loop and mean ± std accumulation.

All four scripts were verified to `py_compile` cleanly, and each was dry-run for
~15-25s before the real runs to confirm: no tracebacks, day loops start at the
correct index (2 for training, 14 for testing), and the CEVD
checkpoint-filename-to-load construction matches what `main_plus.py` will save
(verified by running both `.format(...)` calls with `TRAINING_DAYS = [2, 3, 4,
7, 8, 9, 10, 11]` and diffing the resulting strings — identical).

### Day-index caveat

`TRAINING_DAYS`/test days above rest on an inferred mapping (day-index 1 =
22 March 2016, the paper's validation date), reconstructed because the weekday
count matches the paper's 1+8+5=14 exactly across the 20 available day-files.
This is **not independently confirmed** — no file or code in the repo carries
explicit calendar dates, and a line-count check for a weekday/weekend demand
signal wasn't conclusive either way. Flag this when interpreting final numbers.

### Current runs (as of last update)

Three background processes, all started from `src/` with `.venv/` activated:
```bash
nohup python main_baseline.py > /tmp/baseline_run.log 2>&1 &   # Baseline, 5 test days, no training
nohup python main_vanilla.py -n 500 -d 90 > /tmp/vanilla_train.log 2>&1 &  # NeurADP training, 8 days
nohup python main_plus.py > /tmp/plus_train.log 2>&1 &         # CEVD training, 8 days
```
(`main_plus.py`'s CLI defaults already match — 500 agents, capacity 4, delay 90 —
so it needs no flags; `main_vanilla.py`'s defaults don't, hence `-n 500 -d 90`.)

- **Baseline**: scoring only, no training needed. Fast (~0.2s/step) — expect well
  under an hour for all 5 days. Log: `/tmp/baseline_run.log`.
- **NeurADP training**: `/tmp/vanilla_train.log`. ~8h estimated for all 8 days.
  Once done, its own built-in scoring loop (in the same script, `pre_trained`
  branch — needs to be invoked with pretrained weights loaded, see script) runs
  the 5 test days and prints a final mean ± std.
- **CEVD training**: `/tmp/plus_train.log`. ~8h estimated. Once done, run
  `main_scoring.py` (loads the checkpoint `main_plus.py` just saved, applies the
  λ/ψ combination, scores the 5 test days) for the final CEVD number.

Once all three have full 5-day results: compute mean ± std per method (scripts
already print this), CEVD's % improvement over NeurADP, and compare against the
paper's 85423 ± 2190 / 90286 ± 2108 / 98748.2 ± 2449 / 9.37% — noting the
day-index caveat above.

## How to run things

Baseline (no training needed):
```bash
cd /Users/rupa/sujit/git-root/cevd-ridepooling/src
source ../.venv/bin/activate
python main_baseline.py
```

NeurADP (train, then score — score path is inside the same script's
`pre_trained` branch):
```bash
python main_vanilla.py -n 500 -d 90            # train (8 days)
python main_vanilla.py -n 500 -d 90 -z 1       # score (5 test days), once trained
```

CEVD:
```bash
python main_plus.py       # train (8 days), saves to ../models/
python main_scoring.py    # score (5 test days), loads the checkpoint main_plus.py saved
```

All only need the one venv (`.venv/`) activated — `cplex_bridge.py` transparently
shells out to `.venv-cplex/` for every ILP/LP solve, no manual switching needed.

## Repo state notes

- `.venv/` and `.venv-cplex/` are both gitignored — neither should be committed.
- `src/CentralAgent.py`, `src/Environment.py` have real code changes (swapped
  `model.solve()` for the bridge) — these are intentional and needed regardless
  of machine, since they fix the CPLEX Community Edition size limit, not just the
  arm64/Rosetta problem.
- `src/cplex_bridge.py`, `src/cplex_bridge_worker.py`, `src/main_baseline.py` are
  new files, all needed.
- `src/main_plus.py`, `src/main_vanilla.py`, `src/main_scoring.py` all have real
  config changes now (see "Reproducing the paper's Table 1, row 1" above) — no
  longer just reverted to original committed state.
- The old 2-day-trained `.h5` checkpoints in `models/` (suffix `..._2trained.h5`,
  from the superseded first pass) are stale — the new training runs will save
  checkpoints ending `..._8trained.h5` instead (8 training days now, not 2).
  Fine to delete the old ones once the new ones exist, not urgent.
- CPLEX 20.1 lives at `/Applications/CPLEX_Optimizer201` (Intel-only, no arm64
  support in this base 20.1.0.0 release — a newer CPLEX 22.1.x release would
  reportedly ship native arm64 bindings and avoid needing the bridge entirely,
  if the user's license covers it; not pursued here).
- Full plan, paper cross-check details, and time estimates: `claude_plan.md`
  (repo root).
