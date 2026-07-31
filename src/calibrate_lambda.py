"""Recalibrate CEVD's per-3-hour lambda/alpha constants against our retrained
checkpoint. See calibration_plan.md for the full rationale.

Reuses main_scoring.py's run_epoch directly (imported as a module, so its
`if __name__ == '__main__':` body never runs) rather than duplicating scoring
logic. Runs at reduced scale (100 agents, K=20, preserving the paper's
~5-agents/cluster ratio) on the validation day (day index 1 = 22 March 2016),
which is fully deterministic for is_training=False scoring -- confirmed no
RNG-seeding needed between trials.
"""
import json
import random
import time

import numpy as np
from sklearn.cluster import KMeans

# Force single-threaded TF execution *before* any TF op runs, to eliminate
# floating-point non-determinism from multi-threaded reduction ops (observed:
# identical inputs gave different requests-served counts run-to-run within the
# same process -- see the smoke test below). Must be set before main_scoring
# (which imports tensorflow) is imported.
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import main_scoring as ms
from Environment import NYEnvironment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Oracle import Oracle
from CEVD import PathBasedNN
from Request import Request

# --- Fixed experiment config (mirrors the 500-agent Table-1 run except scale) ---
CAPACITY = 4
PICKUP_DELAY = 90
DECISION_INTERVAL = 60
CALIB_NUM_AGENTS = 100          # reduced scale
CALIB_NUM_CLUSTERS = 20         # preserves ~5 agents/cluster, per paper's ratio
VALIDATION_DAY = 1              # 22 March 2016
TRAIN_NUM_AGENTS_FOR_CHECKPOINT_NAME = 500   # checkpoint trained at 500 agents --
                                              # do NOT substitute CALIB_NUM_AGENTS
TRAINING_DAYS_COUNT = 8
# Reduced fleet alone makes the system severely supply-constrained (measured:
# only ~6.25% of demand served at 100 agents vs ~27-30% at the real 500-agent
# scale), which makes total-requests-served numerically insensitive to which
# specific assignment gets picked -- confirmed empirically: lambda=0 and the
# Team-Temp-1/Uniform-Action anchor gave bit-identical results even though
# per-agent/per-action raw scores clearly differed. Fix: downsample the
# request stream by the same ratio as the fleet reduction, to preserve the
# real per-agent supply/demand balance the paper's calibration implicitly
# assumes.
DOWNSAMPLE = CALIB_NUM_AGENTS / TRAIN_NUM_AGENTS_FOR_CHECKPOINT_NAME

RESULTS_LOG = '/tmp/calibrate_lambda_trials.jsonl'
STATUS_FILE = '/tmp/calibrate_lambda_status.txt'

EXISTING_A =     [-0.65, -0.45, -0.55, -0.6, -0.55, -0.55, -0.6, -0.55]
EXISTING_ALPHA = [7.0, 8.0, 5.0, -10.0, 0.0, -10.0, 10.0, -10.0]
NEUTRAL_A, NEUTRAL_ALPHA = 1.0, 0.0


def status(msg):
    line = "{}: {}".format(time.strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    with open(STATUS_FILE, 'a') as f:
        f.write(line + "\n")


def log_trial(record):
    with open(RESULTS_LOG, 'a') as f:
        f.write(json.dumps(record) + "\n")
    print("TRIAL: {}".format(record), flush=True)


Request.MAX_PICKUP_DELAY = PICKUP_DELAY
Request.MAX_DROPOFF_DELAY = 2 * PICKUP_DELAY

status("Setting up environment: {} agents, K={}".format(CALIB_NUM_AGENTS, CALIB_NUM_CLUSTERS))
envt = NYEnvironment(CALIB_NUM_AGENTS, START_EPOCH=0, STOP_EPOCH=24 * 3600,
                      MAX_CAPACITY=CAPACITY, EPOCH_LENGTH=DECISION_INTERVAL,
                      NUM_CLUSTERS=CALIB_NUM_CLUSTERS)
travel_times = np.array(envt.travel_time)
kmeans = KMeans(n_clusters=CALIB_NUM_CLUSTERS, random_state=0).fit(travel_times)

# Cluster occupancy sanity check (per plan's verification section)
occ = np.bincount(kmeans.labels_, minlength=CALIB_NUM_CLUSTERS)
status("Cluster occupancy (locations per cluster): min={} max={} mean={:.1f}".format(
    occ.min(), occ.max(), occ.mean()))

inter_cluster_distance = np.zeros((CALIB_NUM_CLUSTERS, CALIB_NUM_CLUSTERS), dtype='float32')
all_centers = kmeans.cluster_centers_
all_dist = []
for i in range(CALIB_NUM_CLUSTERS):
    for j in range(CALIB_NUM_CLUSTERS):
        if i - j:
            all_dist.append(np.linalg.norm(all_centers[i] - all_centers[j]))
max_dist = max(all_dist)
min_dist = min(all_dist)
for i in range(CALIB_NUM_CLUSTERS):
    for j in range(CALIB_NUM_CLUSTERS):
        if i - j:
            inter_cluster_distance[i][j] = np.linalg.norm(all_centers[i] - all_centers[j]) / max_dist
        else:
            inter_cluster_distance[i][j] = 0.9 * min_dist / max_dist

envt.cluster_node_dict = kmeans
envt.e = 10

oracle = Oracle(envt)
central_agent = CentralAgent(envt)
value_function = PathBasedNN(envt, load_model_loc=None)

train_file = ('NeurADP+SoftplusPathBasedNN_{}agent_{}capacity_{}delay_{}interval'
              '_vanilla_0sta_24end_2startday_11endday_{}trained.h5').format(
    TRAIN_NUM_AGENTS_FOR_CHECKPOINT_NAME, CAPACITY, PICKUP_DELAY,
    DECISION_INTERVAL, TRAINING_DAYS_COUNT)
status("Loading checkpoint: {}".format(train_file))
value_function.model.load_weights('../models/' + train_file)

ms.log = {}
random.seed(0)

status("Running pickup-average pre-pass for validation day {}".format(VALIDATION_DAY))
pickup_avg = ms.run_epoch(envt, oracle, central_agent, kmeans, value_function,
                          VALIDATION_DAY, is_training=False,
                          inter_cluster_distance=inter_cluster_distance,
                          lamb=None, cont=True,
                          a_arr=[0.0] * 8, alpha_arr=[0.0] * 8,
                          downsample=DOWNSAMPLE)
status("Pickup average pre-pass done: {}".format(pickup_avg))


def evaluate(a_arr, alpha_arr, label=""):
    """Run one full validation-day scoring pass with the given 16 constants."""
    ms.log = {}
    ms.day = VALIDATION_DAY
    # envt is reused across trials (built once for speed) -- reset the state
    # that would otherwise leak between calls: recent_request_history feeds
    # rebalancing target sampling (Environment.py:125), and that sampling
    # consumes from the global `random` module state, so both must be reset
    # for trials to be truly comparable / reproducible.
    envt.recent_request_history.clear()
    random.seed(0)
    initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training=False)
    agents = [LearningAgent(i, s) for i, s in enumerate(initial_states)]
    t0 = time.perf_counter()
    served = ms.run_epoch(envt, oracle, central_agent, kmeans, value_function,
                           VALIDATION_DAY, is_training=False,
                           agents_predefined=agents,
                           inter_cluster_distance=inter_cluster_distance,
                           predicted_demand=[0], pickup_avg=pickup_avg,
                           a_arr=list(a_arr), alpha_arr=list(alpha_arr),
                           downsample=DOWNSAMPLE)
    elapsed = time.perf_counter() - t0
    record = {"label": label, "a_arr": list(a_arr), "alpha_arr": list(alpha_arr),
              "served": served, "elapsed_s": elapsed}
    log_trial(record)
    return served


# --- Determinism smoke test: repeat one trial, expect identical result ---
status("Smoke test: repeating one trial to check determinism")
s1 = evaluate(EXISTING_A, EXISTING_ALPHA, label="smoke-1")
s2 = evaluate(EXISTING_A, EXISTING_ALPHA, label="smoke-2")
if s1 != s2:
    status("WARNING: determinism check failed ({} != {}) -- results may not be reproducible".format(s1, s2))
else:
    status("Determinism check OK: {} == {}".format(s1, s2))

# --- Step 0: falsification checks ---
status("Step 0: falsification checks")
served_lamb0 = evaluate([0.0] * 8, EXISTING_ALPHA, label="step0-lambda0")
served_anchor = evaluate([1.0] * 8, [0.0] * 8, label="step0-anchor")
status("Step 0 results: lambda=0 (NeurADP-equivalent) served={}, "
       "Team-Temp-1 & Uniform-Action anchor served={}".format(served_lamb0, served_anchor))

if served_lamb0 <= 0:
    status("ABORT: lambda=0 baseline served 0 or negative requests -- harness is broken, "
           "not a calibration problem. Stopping before Step 1.")
    raise SystemExit(1)

improvement = 100.0 * (served_anchor - served_lamb0) / served_lamb0
status("Anchor improvement over lambda=0 baseline: {:.2f}%".format(improvement))
if improvement < 1.0:
    status("ABORT: anchor (Team-Temp-1 + Uniform-Action) did not meaningfully beat "
           "the lambda=0 baseline (got {:.2f}%, expected roughly +4.5% per the paper's "
           "weaker ablation). This suggests a structural issue, not just miscalibrated "
           "constants -- see calibration_plan.md Step 0 investigation list. Stopping "
           "before Step 1.".format(improvement))
    raise SystemExit(1)

status("Step 0 gate passed. Proceeding to Step 1 (per-bucket search).")

# --- Step 1: sequential per-bucket search, temporal order, nested a/alpha sweep ---
locked_a = [None] * 8
locked_alpha = [None] * 8


def build_arrs(r, a_r, alpha_r):
    a = [locked_a[i] if i < r else (a_r if i == r else NEUTRAL_A) for i in range(8)]
    al = [locked_alpha[i] if i < r else (alpha_r if i == r else NEUTRAL_ALPHA) for i in range(8)]
    return a, al


for r in range(8):
    status("Bucket {}/7: starting a-sweep".format(r))
    a_candidates = sorted({EXISTING_A[r], 1.0, 0.0,
                           EXISTING_A[r] - 0.3, EXISTING_A[r] - 0.15,
                           EXISTING_A[r] + 0.15, EXISTING_A[r] + 0.3})
    best_a, best_score = EXISTING_A[r], None
    for a_r in a_candidates:
        a, al = build_arrs(r, a_r, EXISTING_ALPHA[r])
        score = evaluate(a, al, label="bucket{}-a-sweep-a={}".format(r, a_r))
        if best_score is None or score > best_score:
            best_a, best_score = a_r, score

    status("Bucket {}: a-sweep winner a={} (served={}), starting alpha-sweep".format(
        r, best_a, best_score))
    alpha_candidates = sorted({EXISTING_ALPHA[r], 0.0,
                              EXISTING_ALPHA[r] - 6, EXISTING_ALPHA[r] - 3,
                              EXISTING_ALPHA[r] + 3, EXISTING_ALPHA[r] + 6})
    best_alpha, best_score2 = EXISTING_ALPHA[r], best_score
    for alpha_r in alpha_candidates:
        a, al = build_arrs(r, best_a, alpha_r)
        score = evaluate(a, al, label="bucket{}-alpha-sweep-alpha={}".format(r, alpha_r))
        if score > best_score2:
            best_alpha, best_score2 = alpha_r, score

    locked_a[r], locked_alpha[r] = best_a, best_alpha
    status("Bucket {} LOCKED: a={}, alpha={}, served={}".format(r, best_a, best_alpha, best_score2))

status("Search complete.")
status("Final a_arr: {}".format(locked_a))
status("Final alpha_arr: {}".format(locked_alpha))

with open('/tmp/calibrate_lambda_final.json', 'w') as f:
    json.dump({"a_arr": locked_a, "alpha_arr": locked_alpha}, f, indent=2)
status("Wrote final constants to /tmp/calibrate_lambda_final.json")
