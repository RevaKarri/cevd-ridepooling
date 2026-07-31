# Recalibrating CEVD's λ/ψ constants against our retrained checkpoint

Status: implementation-ready plan. No code has been changed or run to produce this
document. Full paper-research backing this plan lives at
`/Users/rupa/.claude/plans/majestic-plotting-scone-agent-ae4888d155f853698.md`;
this document only restates what's load-bearing for implementation and adds the
code-level details discovered by re-reading `main_scoring.py` and `CEVD.py` line
by line while drafting this plan.

## 1. Context

We reproduced Table 1, row 1 of "On Sustainable Ride Pooling Through Conditional
Expected Value Decomposition" (Bose, Jiang, Varakantham, Ge; ECAI 2023) — 5-day
average Requests Served at 500 vehicles / capacity 4 / pickup delay 90s / 60s
decision interval:

| | Ours | Paper |
|---|---|---|
| Baseline | 85899.40 ± 2896.87 | 85423 ± 2190 |
| NeurADP | 91094.00 ± 3241.01 | 90286 ± 2108 |
| CEVD | **85764.00 ± 2865.36** | 98748.2 ± 2449 (+9.37% over NeurADP) |

Baseline and NeurADP match closely. CEVD does not — it lands *below* both
Baseline and NeurADP instead of ~9% above NeurADP.

**Root cause.** `main_scoring.py::run_epoch` applies CEVD's neighbor-value
combination (the paper's Eq. 4) at scoring time using hardcoded per-3-hour-bucket
constants, defined inside `run_epoch` itself (not at module scope — see §4.1):

```python
# main_scoring.py, inside run_epoch(), lines ~73-86
b_arr = [0, 0, 0, 0, 0, 0, 0, 0]
if(args.numagents == 500 and args.capacity == 4 and args.pickupdelay == 90):
    a_arr = [-0.65, -0.45, -0.55, -0.6, -0.55, -0.55, -0.6, -0.55]
    alpha_arr = [7.0, 8.0, 5.0, -10.0, 0.0, -10.0, 10.0, -10.0]
elif(...):  # three more (capacity, pickupdelay) combinations
    ...
```

Every 180 decision steps (3 simulated hours — one bucket), `run_epoch` advances
`r` and sets `a, b, alpha = a_arr[r], b_arr[r], alpha_arr[r]`, then
`lamb = a * exp(-b * pickup_avg)` and `P[i][j] = exp(alpha * inter_cluster_distance[i][j])`,
which feed into `CEVD.py::NeuralNetworkBased.get_value`'s neighbor-combination
term (`V_i_hat = (term1 + lamb[count3]*term2) / (1 + lamb[count3])`, line 372).

These 16 numbers (8 `a` + 8 `alpha`) were almost certainly fit by the original
authors against *their* trained value network. Our checkpoint (`main_plus.py` /
`NeurADPplus`, trained by plain individual TD) is a different function, so the
same constants pull the neighbor combination in a different, apparently harmful,
direction for us.

**Why we can't just fix this by training.** `CEVD.py`'s `update()` is the only
code path that would jointly fit `θ` (the value network) together with the
`λ`/`ψ` combination (Algorithm 1 line 15 / Eq. 8 in the paper). It has a live
bug: it calls `self.get_value(experiences=[experience], P=P, cluster_info=cluster_info, network=self.model)`
(line 443) without passing `lamb`, so `get_value`'s default `lamb=1` (a scalar)
is used — but `get_value`'s CEVD branch does `lamb[count3]` (line 372), which
requires an indexable array and will throw. This path has never been exercised:
`main_scoring.py`'s `__main__` hardcodes `pre_trained = True` and never calls
`run_epoch(..., is_training=True)`. So it's dead, broken code, not a shortcut —
reimplementing real joint training is a separate, larger undertaking, out of
scope here. This plan is a **post-hoc calibration** of the existing 16 constants
against our checkpoint, evaluated purely at scoring time.

## 2. What the paper actually licenses (summary; full detail in the linked doc)

1. **λ is not restricted to [0,1].** Theorem 1 defines team temperatures as
   arbitrary real-valued (`λ^i : S → ℝ`); Eq. 4 is an *affine*, not convex,
   combination. The existing negative `a_arr` values are within the paper's
   hypothesis space, not a bug — keep negative territory in the search grid.
2. **λ has no dependence on cluster state in the paper.** Section 5: "a
   piecewise linear function consisting of 8 constant pieces, 1 for each 3 hour
   interval" — piecewise-*constant* in time only. The repo's
   `lamb = a * exp(-b * pickup_avg)` is a strict generalization, currently
   neutralized by `b_arr ≡ 0`. **Keep `b` pinned at 0** throughout calibration —
   don't introduce degrees of freedom the paper doesn't describe. This confirms
   the search space is exactly 16 scalars (8 `a`, 8 `alpha`), not 24.
3. **ψ/alpha's functional form matches the paper exactly**: piecewise-linear
   (i.e. piecewise-constant-per-bucket) in time, `P^{j|i}_ψ(g|·) ∝ e^{ψt ×
   pickup-distance}`. Nothing to change conceptually.
4. **Two free, checkpoint-agnostic anchor points come straight from the
   ablation study (Sec 6.3)**, and should seed the search grid:
   - **Team-Temp-1**: `λ^k(s_t) = 1` for every bucket → **+4.58%** over NeurADP
     in the paper.
   - **Uniform-Action**: `P^{kj|ki}(g|·) = 1/|F^t_kj|` for every `g` → exactly
     what `alpha ≡ 0` gives (`exp(0·distance) = 1`, normalizes to uniform) →
     **+4.67%** over NeurADP in the paper.
   - `λ ≡ 0` recovers NeurADP exactly (paper states this directly) — a free
     self-consistency check on our own harness.
   These are not the *optimal* Table-1 constants (those 16 numbers are never
   published), but they're real, validated points on the same objective surface
   we're searching, and cost nothing to include.
5. **Cluster-to-fleet ratio is load-bearing, not incidental.** The paper holds
   agents/cluster ≈ 5 across fleet sizes (K=100/150/200 for 500/750/1000
   vehicles). Eq. 4's `|Ck|-1` denominator means the neighbor-pool size is
   exactly what `λ`/`ψ` weight. A reduced-scale calibration run must rescale K
   to preserve this ratio, or the neighbor term becomes vacuous and the search
   will "learn" that `a`/`alpha` don't matter — silently defeating the exercise.
6. **Structural ceiling** (state this honestly up front, don't let it surprise
   anyone at the end): our `Qθ` was never trained jointly with the CEVD
   combination, unlike Eq. 8's joint loss. Post-hoc calibration finds the best
   affine combination of a `Qθ` that wasn't fit with this combination in mind.
   Landing around the ablation-level improvement (~+4-5% over NeurADP) rather
   than the full +9.37% would be an expected, reasonable outcome, not a failure
   of this procedure.

## 3. Approach

### 3.1 Step 0 — cheap falsification check (run this before any search)

At reduced scale (see §3.3 for exact numbers), on the validation day, run two
parameter-free configurations against our checkpoint:

1. `a_arr = [0]*8` (→ `λ≡0` for all buckets) — should reproduce a
   requests-served number close to our already-measured NeurADP result (91094,
   at reduced scale it'll differ in absolute terms but should track NeurADP's
   at-that-scale number, not CEVD's). This validates that the scoring harness
   itself, run through `run_epoch`'s CEVD branch with `λ=0`, degrades correctly
   to plain NeurADP behavior.
2. `a_arr = [1]*8`, `alpha_arr = [0]*8` (→ Team-Temp-1 AND Uniform-Action
   combined) — should beat the reduced-scale NeurADP-equivalent number by
   roughly the ballpark of the *weaker* reported ablation, ~+4.5%, at minimum
   (the paper doesn't report this exact combination, but both ablations
   individually clear NeurADP, so the combination should too).

**Hard gate**: if neither check lands near its expected ballpark, stop before
running the coordinate search — investigate first:
- Whether `CEVD.py`'s `get_value` divides the neighbor-average by the *current*
  cluster occupancy (`|Ck|-1`, i.e. `len(all_marginals[f]) - 1`, line 369) as
  Eq. 4 specifies, not by `K` or a fixed constant. (Read confirms it does
  divide by `len(all_marginals[f]) - 1`, which is the number of cluster-mates
  with a request-cluster overlapping `f` — this is a legitimate, if slightly
  narrower, reading of `|Ck|-1`; worth a second look if Step 0 fails, but not
  something this plan changes.)
- Whether the reduced-scale K actually preserves non-degenerate cluster
  occupancy (§3.3) — a search can't fix a structural bug or a degenerate
  neighbor pool, and either would waste the entire search budget compensating
  for something a parameter choice can't repair.

### 3.2 Step 1 — coordinate-wise search, single temporal pass, bucket-local nested sweep

The MMDP's transitions are causal: bucket `r`'s decisions affect the state
*entering* buckets `r+1..7` (vehicle positions, in-progress trips), never
buckets before `r`. So there's no need for multi-pass coordinate descent to
reach a fixed point — sweep buckets **once, in order 0→7**, locking in each
bucket's winner before moving to the next.

For bucket `r`:
- Buckets `< r`: already locked to their chosen winners.
- Buckets `> r` (not yet tuned): held at the neutral paper anchor, `a=1,
  alpha=0` — a real, validated combination (§2.4), not an arbitrary filler.
- Bucket `r` itself: search in two nested passes to bound cost (full 2D grid
  per bucket would be ~30-40 evals × 8 buckets; nesting cuts this to ~12):
  1. **Sweep `a`** with `alpha` held at bucket `r`'s existing hardcoded value.
     Candidate set: `{existing a_r (warm start), 1.0, 0.0, a_r - 0.3, a_r -
     0.15, a_r + 0.15, a_r + 0.3}` — 7 candidates. Keep the best-scoring `a`.
  2. **Sweep `alpha`** with `a` fixed to the winner from step 1. Candidate set:
     `{existing alpha_r (warm start), 0.0, alpha_r - 6, alpha_r - 3, alpha_r +
     3, alpha_r + 6}` — 6 candidates.
  - Total per bucket: 13 evaluations (7 + 6, with one candidate — the a-sweep
    winner re-scored implicitly as the alpha-sweep's baseline — effectively 12
    net new evaluations). Across 8 buckets: **~96-104 evaluations**, plus
    Step 0's 2.
- `b` stays pinned at `[0]*8` throughout — not part of the search space (§2.2).

This is a pragmatic substitute for Eq. 8's joint gradient descent, justified by:
16 independently-interpretable, physically-meaningful scalars, each anchored by
a paper-validated reference point, evaluated on a metric (full-day requests
served) with no evidence of a pathological loss surface. A generic
metaheuristic (Bayesian optimization, CMA-ES) is not warranted here.

### 3.3 Scale for the search

Use a reduced fleet, but **do not reuse the 500-agent K=100 default**. The
paper holds agents/cluster ≈ 5 across fleet sizes (500/100, 750/150, 1000/200).
Use **~100 agents with K≈20** clusters to preserve that ratio (100/20 = 5,
matching the paper's ratio). `NUM_CLUSTERS` only feeds the KMeans clustering of
map locations (used purely for the CEVD neighbor bookkeeping — `P`,
`cluster_info`, `request_clusters`); it does not affect the value network's
inputs at all (`_format_input_batch` never references cluster id), so this is
safe to vary independently of the checkpoint, which is agent-count-agnostic by
construction (no fixed-size agent dimension in the Keras model).

Evaluate on the **validation day, 22 March 2016**. Confirmed from the repo's
day-indexing convention: `TEST_DAYS_TO_RUN = [14, 15, 16, 17, 18]` maps to
4-8 April 2016 (5 test days, matching the paper), and `TRAINING_DAYS = [2, 3,
4, 7, 8, 9, 10, 11]` (8 weekdays, skipping the weekend at indices 5-6) maps to
23 March-1 April 2016 (matching the paper's stated training window) — so day
index `N` = 21 March 2016 + `N` days, and **day index 1 = 22 March 2016**, the
paper's validation day, distinct from both the training days (2-11) and the
test days (14-18). Confirmed the data file `test_flow_5000_1.txt` exists under
`data/ny/files_60sec/`.

**Determinism**: `Environment.get_initial_states(..., is_training=False)`
returns a fixed prefix of `self.initial_zones` (not randomized), and the ILP's
noise term is zeroed when not training. Candidate evaluations are fully
reproducible given the same constants, checkpoint, and day — no seeding needed
across trials, and repeated trials with identical constants should return
bit-identical requests-served counts (useful as a smoke-test invariant).

**Runtime budget**: not benchmarked fresh for this plan — before committing to
the full ~100-evaluation search, do a short dry run (2-3 trials) at the chosen
scale and time it directly, then extrapolate. As a rough planning anchor, if a
100-agent validation-day pass takes on the order of minutes rather than hours
(it should, being 5x smaller than the 500-agent runs already completed), the
full search is a same-day, unattended background job; if it's much slower than
that, cut the per-bucket candidate counts in half before running all 8 buckets
serially. Don't guess a total-hours figure in this doc that hasn't been
measured on this run — measure it once, then decide the grid density from that.

## 4. Implementation

New script: `src/calibrate_lambda.py`. Reuses `main_scoring.py` directly rather
than duplicating scoring logic — `main_scoring.py` guards its script body with
`if __name__ == '__main__':`, so `import main_scoring as ms` only pulls in the
`run_epoch` function and the module's top-level imports; no side effects fire.

### 4.1 Required one-time patch to `main_scoring.py` (do this first)

This is the single most important implementation detail, and it corrects an
assumption worth stating precisely: `a_arr`/`b_arr`/`alpha_arr` are **not**
already module-level globals — they are **local variables computed inside
`run_epoch`** (lines ~73-86), derived from the module-level `args` Namespace
(built by `argparse` in `__main__`) via an `if/elif` ladder keyed on
`args.numagents`/`args.capacity`/`args.pickupdelay`. Two consequences:

- Calling `run_epoch` at a reduced agent count (e.g. 100) as-is will hit
  **none** of the four `elif` branches, so `a_arr`/`alpha_arr` are never
  assigned, and the function raises `NameError` the first time it reaches
  `a = a_arr[r]`.
- There is currently no seam for a caller to inject candidate `a_arr`/
  `alpha_arr` values into `run_epoch` at all — the CLI's `-a`/`-b`/`-l`
  arguments (`args.a/args.b/args.alpha`) are accepted as `run_epoch` keyword
  parameters (`a=None, b=None, alpha=None`) but are immediately clobbered by
  the hardcoded ladder the very first time `t % 180 == 0` fires, which happens
  at `t=0` — so they're already vestigial/dead for real runs, consistent with
  the observed root cause.

Fix: make the three arrays genuine optional parameters that shadow the ladder
when supplied, defaulting to `None` so **all existing behavior (the CLI script,
the eventual full-scale confirmation run) is unchanged byte-for-byte** when they
aren't passed:

```python
def run_epoch(envt, oracle, central_agent, kmeans, value_function, DAY,
              is_training, agents_predefined=None, TRAINING_FREQUENCY: int=1,
              inter_cluster_distance=None, lamb=1, decay=0.8, cont=False,
              pickup_avg=None, a=None, b=None, alpha=None,
              predicted_demand=None,
              a_arr=None, b_arr=None, alpha_arr=None):   # <-- new, additive
    ...
    if b_arr is None:
        b_arr = [0, 0, 0, 0, 0, 0, 0, 0]
    if a_arr is None or alpha_arr is None:
        # Estimated lambda, alpha values (fallback: only reached when the
        # caller doesn't supply overrides — preserves existing CLI behavior)
        if(args.numagents == 500 and args.capacity == 4 and args.pickupdelay == 90):
            a_arr = [-0.65, -0.45, -0.55, -0.6, -0.55, -0.55, -0.6, -0.55]
            alpha_arr = [7.0, 8.0, 5.0, -10.0, 0.0, -10.0, 10.0, -10.0]
        elif(args.numagents == 500 and args.capacity == 5 and args.pickupdelay == 90):
            ...  # unchanged, all four existing elif branches kept verbatim
```

This is the only change needed to `main_scoring.py` before the search; it's
purely additive (new keyword args, default `None`, existing call sites
untouched) and carries zero risk to the already-validated Baseline/NeurADP/CEVD
reproduction runs.

Two more globals `run_epoch` depends on that a library caller must set, both
easy to miss because they aren't parameters:

- **`log`** (a module-level `dict`): referenced and mutated throughout
  `run_epoch` (e.g. `log['total_day_{}_time_{}'...] = ...`, lines ~228-230 and
  ~244-246). Must exist (`ms.log = {}`) before any call with
  `is_training=False`, or `run_epoch` raises `NameError` on first use.
- **`day`** (a module-level variable, lowercase): `run_epoch`'s own logging
  code at lines ~228-230 and ~244-246 uses the bare name `day`, **not** its own
  `DAY` parameter, to build log keys (`'total_day_{}_time_{}'.format(day,
  ...)`). In the CLI script this resolves because `__main__`'s `for day in
  TEST_DAYS_TO_RUN:` loop leaves `day` as a live module global by the time
  `run_epoch` runs. A caller that imports `run_epoch` directly must set
  `ms.day = <the day index being scored>` before each call (in addition to
  passing the same value as the `DAY` argument), or hit the same `NameError`.
  This is a pre-existing quirk in `main_scoring.py`, not something to "fix" as
  part of this plan — just work around it from the calibration script.

Neither of these two globals needs touching in `main_scoring.py` itself; they
just need to be set from `calibrate_lambda.py` before each scoring call.

One more latent-but-currently-harmless quirk, noted so it isn't accidentally
"fixed" mid-calibration and doesn't need touching: the `kmeans_loc` / `isfile`
branch in `__main__` prints "Using Saved K-Means" but never actually assigns
`kmeans` in that branch (the `pickle.load` line is commented out) — it only
works today because no `*numclusters.sav` file exists anywhere in the repo, so
the `else` branch (fresh `KMeans(n_clusters=K, random_state=0).fit(travel_times)`)
always runs in practice. `calibrate_lambda.py` should just always fit fresh
KMeans directly (mirroring the `else` branch), never the `isfile` branch.

### 4.2 `calibrate_lambda.py` structure

```python
import main_scoring as ms
from Environment import NYEnvironment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Oracle import Oracle
from CEVD import PathBasedNN
from sklearn.cluster import KMeans
import numpy as np

# --- Fixed experiment config (mirrors the 500-agent Table-1 run except scale) ---
CAPACITY = 4
PICKUP_DELAY = 90
DECISION_INTERVAL = 60
CALIB_NUM_AGENTS = 100          # reduced scale
CALIB_NUM_CLUSTERS = 20         # preserves ~5 agents/cluster, per paper's ratio
VALIDATION_DAY = 1              # 22 March 2016 (see §3.3 derivation)
TRAIN_NUM_AGENTS_FOR_CHECKPOINT_NAME = 500   # checkpoint was trained at 500 agents —
                                              # do NOT substitute CALIB_NUM_AGENTS here
TRAINING_DAYS_COUNT = 8          # matches TRAINING_DAYS in main_scoring.py's __main__

Request.MAX_PICKUP_DELAY = PICKUP_DELAY
Request.MAX_DROPOFF_DELAY = 2 * PICKUP_DELAY

# --- One-time setup (build once, reuse across every trial) ---
envt = NYEnvironment(CALIB_NUM_AGENTS, START_EPOCH=0, STOP_EPOCH=24*3600,
                     MAX_CAPACITY=CAPACITY, EPOCH_LENGTH=DECISION_INTERVAL,
                     NUM_CLUSTERS=CALIB_NUM_CLUSTERS)
travel_times = np.array(envt.travel_time)
kmeans = KMeans(n_clusters=CALIB_NUM_CLUSTERS, random_state=0).fit(travel_times)
# ... build inter_cluster_distance exactly as main_scoring.py's __main__ does
# (normalize by max pairwise center distance, 0.9*min_dist/max_dist on the diagonal)
envt.cluster_node_dict = kmeans
envt.e = 10  # matches --samplingperagent default

oracle = Oracle(envt)
central_agent = CentralAgent(envt)
value_function = PathBasedNN(envt, load_model_loc=None)
train_file = ('NeurADP+SoftplusPathBasedNN_{}agent_{}capacity_{}delay_{}interval'
              '_vanilla_0sta_24end_2startday_11endday_{}trained.h5').format(
    TRAIN_NUM_AGENTS_FOR_CHECKPOINT_NAME, CAPACITY, PICKUP_DELAY,
    DECISION_INTERVAL, TRAINING_DAYS_COUNT)
value_function.model.load_weights('../models/' + train_file)

ms.log = {}   # required module global (see §4.1)

# Pickup-average pre-pass (cont=True), computed once for the validation day —
# depends only on request data, not on a_arr/alpha_arr, so it's shared across
# every trial. Must still pass dummy a_arr/alpha_arr since run_epoch evaluates
# the assignment block unconditionally before checking `cont`.
pickup_avg = ms.run_epoch(envt, oracle, central_agent, kmeans, value_function,
                          VALIDATION_DAY, is_training=False,
                          inter_cluster_distance=inter_cluster_distance,
                          lamb=None, cont=True,
                          a_arr=[0]*8, alpha_arr=[0]*8)

def evaluate(a_arr, alpha_arr):
    """Run one full validation-day scoring pass with the given 16 constants."""
    ms.day = VALIDATION_DAY   # required module global (see §4.1)
    initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training=False)
    agents = [LearningAgent(i, s) for i, s in enumerate(initial_states)]
    served = ms.run_epoch(envt, oracle, central_agent, kmeans, value_function,
                          VALIDATION_DAY, is_training=False,
                          agents_predefined=agents,
                          inter_cluster_distance=inter_cluster_distance,
                          predicted_demand=[0], pickup_avg=pickup_avg,
                          a_arr=a_arr, alpha_arr=alpha_arr)
    return served

# --- Step 0: falsification checks ---
EXISTING_A =     [-0.65, -0.45, -0.55, -0.6, -0.55, -0.55, -0.6, -0.55]
EXISTING_ALPHA = [7.0, 8.0, 5.0, -10.0, 0.0, -10.0, 10.0, -10.0]

served_lamb0 = evaluate([0.0]*8, EXISTING_ALPHA)     # -> should ~ NeurADP-equivalent
served_anchor = evaluate([1.0]*8, [0.0]*8)           # -> should beat it by ~4.5%+
log(...)  # print both clearly; abort with a clear message and do NOT proceed
          # to Step 1 if served_anchor doesn't clear served_lamb0 by a healthy margin

# --- Step 1: sequential per-bucket search ---
NEUTRAL_A, NEUTRAL_ALPHA = 1.0, 0.0
locked_a = [None]*8
locked_alpha = [None]*8

for r in range(8):
    def build_arrs(a_r, alpha_r):
        a = [locked_a[i] if i < r else (a_r if i == r else NEUTRAL_A) for i in range(8)]
        al = [locked_alpha[i] if i < r else (alpha_r if i == r else NEUTRAL_ALPHA) for i in range(8)]
        return a, al

    # Pass 1: sweep a, alpha held at existing value for bucket r
    a_candidates = sorted({EXISTING_A[r], 1.0, 0.0,
                            EXISTING_A[r]-0.3, EXISTING_A[r]-0.15,
                            EXISTING_A[r]+0.15, EXISTING_A[r]+0.3})
    best_a, best_score = EXISTING_A[r], -1
    for a_r in a_candidates:
        a, al = build_arrs(a_r, EXISTING_ALPHA[r])
        score = evaluate(a, al)
        log_trial(r, "a-sweep", a_r, EXISTING_ALPHA[r], score)
        if score > best_score:
            best_a, best_score = a_r, score

    # Pass 2: sweep alpha, a fixed to best_a
    alpha_candidates = sorted({EXISTING_ALPHA[r], 0.0,
                                EXISTING_ALPHA[r]-6, EXISTING_ALPHA[r]-3,
                                EXISTING_ALPHA[r]+3, EXISTING_ALPHA[r]+6})
    best_alpha, best_score2 = EXISTING_ALPHA[r], best_score
    for alpha_r in alpha_candidates:
        a, al = build_arrs(best_a, alpha_r)
        score = evaluate(a, al)
        log_trial(r, "alpha-sweep", best_a, alpha_r, score)
        if score > best_score2:
            best_alpha, best_score2 = alpha_r, score

    locked_a[r], locked_alpha[r] = best_a, best_alpha
    print(f"Bucket {r} locked: a={best_a}, alpha={best_alpha}, score={best_score2}")

print("Final a_arr:", locked_a)
print("Final alpha_arr:", locked_alpha)
```

Notes on the sketch above:
- `build_arrs` is the mechanism enforcing §3.2's rule: buckets `< r` use their
  already-locked winners, bucket `r` uses the candidate under test, buckets
  `> r` use the neutral paper anchor.
- Every trial's `(bucket, sweep-phase, a, alpha, score)` should be logged (to
  stdout and/or a small JSON/CSV file) for later inspection — this is cheap and
  makes the search auditable and resumable if interrupted.
- Because evaluation is deterministic (§3.3), no repeated trials or averaging
  are needed per candidate — one evaluation per candidate is sufficient, and a
  repeated call with identical constants is a good smoke test to run once at
  the very start.

### 4.3 After the search

1. Print the final 8-length `a_arr`/`alpha_arr` clearly; do not auto-apply
   mid-search — review the numbers by hand first (sanity-check sign patterns,
   magnitude relative to the existing constants, and whether any bucket's
   winner is sitting at a grid boundary, which would suggest widening that
   bucket's candidate range and re-searching just that bucket).
2. Patch the winning 16 constants directly into `main_scoring.py`'s existing
   `if(args.numagents == 500 and args.capacity == 4 and args.pickupdelay == 90):`
   branch (the literal `a_arr =` / `alpha_arr =` lines, currently
   `[-0.65, -0.45, ...]` / `[7.0, 8.0, ...]`) — replacing the old, apparently
   miscalibrated values with the new ones. Leave the other three `elif`
   branches (capacity=5, delay=120, delay=150) untouched — they're out of
   scope for this Table-1-row-1 recalibration. The additive override
   parameters from §4.1 stay in the code permanently (harmless, useful for any
   future recalibration) but are not exercised by the plain CLI invocation.
3. Re-run the full 500-agent, 5-test-day CEVD scoring pass (`python
   main_scoring.py -n 500 -q 4 -d 90 -t 60`, matching how Baseline/NeurADP/CEVD
   were already produced) to get the final, confirmed CEVD number.

## 5. Verification

- **Pre-search smoke test**: with the §4.1 patch applied, run
  `calibrate_lambda.py` for 1-2 trials only (not the full search) and confirm:
  no tracebacks; `Max lamb`/`Min lamb`/`Max P`/`Min P` (already printed by
  `run_epoch`) actually change between trials with different `a_arr`/
  `alpha_arr`; repeating one trial with identical constants reproduces the
  identical requests-served number (determinism check); the K=20/100-agent
  setup produces non-degenerate cluster occupancy (spot-check `cluster_info`
  list lengths aren't mostly 0-1).
- **Step 0 gate**: both falsification checks (§3.1) must land near their
  expected ballparks before Step 1 runs at all. Treat this as a hard abort
  condition in the script, not just a printed warning.
- **Per-bucket sanity**: each bucket's locked-in winner must score at least as
  well as every candidate in that bucket's own grid (true by construction,
  since the winner is selected as the arg-max over the grid) — log this
  explicitly so it's visible in the trial log, as a basic correctness check on
  the search loop itself.
- **Final confirmation**: after patching `main_scoring.py` and re-running the
  full 500-agent, 5-test-day pass, compare against:
  - Paper's CEVD number: 98748.2 ± 2449 (+9.37% over NeurADP).
  - Our own Baseline (85899.40 ± 2896.87) and NeurADP (91094.00 ± 3241.01) —
    the correct comparison baseline per §2.6 and per how Table 1 itself defines
    "% improvement."
  - The paper's two ablation-level reference points (+4.58%, +4.67% over
    NeurADP) as calibration sanity anchors. Landing in that range rather than
    at the full +9.37% is an expected, acceptable outcome given the structural
    ceiling in §2.6 (our `Qθ` was never trained jointly with the CEVD
    combination) — report the achieved number honestly against both references
    rather than treating anything short of +9.37% as a failure of this
    procedure.
  - If the recalibrated CEVD number is *still* at or below NeurADP, that's a
    signal the Step 0 gate should have caught something (or was skipped) —
    revisit §3.1's investigation list before concluding calibration alone
    can't help.

## 6. Summary of concrete deliverables

1. Small, additive patch to `main_scoring.py::run_epoch` (§4.1): three new
   optional keyword parameters (`a_arr`, `b_arr`, `alpha_arr`) that shadow the
   existing hardcoded ladder when supplied; zero behavior change when omitted.
2. New script `src/calibrate_lambda.py` (§4.2): one-time setup (100 agents,
   K=20, real checkpoint loaded), Step 0 falsification gate, Step 1 sequential
   per-bucket nested search (a-sweep then alpha-sweep, 8 buckets, ~100 total
   evaluations), full trial logging, final printed 16-constant result.
3. Manual review of the winning constants, then a second small patch to
   `main_scoring.py`'s `numagents==500/capacity==4/pickupdelay==90` branch
   (literal `a_arr`/`alpha_arr` values only).
4. Full 500-agent, 5-test-day confirmation re-run of `main_scoring.py`,
   reported against Baseline/NeurADP/paper's CEVD/paper's two ablation anchors.
