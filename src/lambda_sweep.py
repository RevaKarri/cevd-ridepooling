"""Diagnostic lambda-response sweep after the Step 0 gate abort.

Maps requests-served as a function of a single uniform lambda (all 8 buckets
identical), with alpha=0 (uniform P) except where noted, on the validation day
at reduced scale. Purpose: determine whether ANY lambda beats lambda=0 on our
independently-TD-trained checkpoint. If the curve peaks at lambda=0, post-hoc
calibration cannot recover the paper's CEVD gain (joint training would be
required), and the search is conclusively not worth running.
"""
import json
import random
import time

import numpy as np
from sklearn.cluster import KMeans

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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--numagents', type=int, default=100)
parser.add_argument('-l', '--lambdas', type=str, default=None,
                    help='comma-separated lambda values (alpha=0) overriding the default grid')
parser.add_argument('-p', '--profile', type=str, default=None,
                    help='comma-separated 8 lambda values, one per 3-hour bucket (alpha=0), '
                         'evaluated as a single non-uniform candidate instead of the uniform grid')
sweep_args = parser.parse_args()
if sweep_args.profile:
    PROFILE = [float(x) for x in sweep_args.profile.split(',')]
    assert len(PROFILE) == 8, "profile must have exactly 8 values (one per 3h bucket)"

CAPACITY = 4
PICKUP_DELAY = 90
DECISION_INTERVAL = 60
CALIB_NUM_AGENTS = sweep_args.numagents
# Preserve the paper's ~5 agents/cluster ratio (K=100 at 500 vehicles) and
# scale demand by the same ratio as the fleet, so per-agent supply/demand
# balance matches the real 500-agent regime.
CALIB_NUM_CLUSTERS = max(2, round(CALIB_NUM_AGENTS / 5))
VALIDATION_DAY = 1
DOWNSAMPLE = CALIB_NUM_AGENTS / 500.0

FILE_TAG = '{}agents'.format(CALIB_NUM_AGENTS) + ('_ext' if sweep_args.lambdas else '') + ('_profile' if sweep_args.profile else '')
RESULTS_LOG = '/tmp/lambda_sweep_trials_{}.jsonl'.format(FILE_TAG)
STATUS_FILE = '/tmp/lambda_sweep_status_{}.txt'.format(FILE_TAG)


def status(msg):
    line = "{}: {}".format(time.strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    with open(STATUS_FILE, 'a') as f:
        f.write(line + "\n")


Request.MAX_PICKUP_DELAY = PICKUP_DELAY
Request.MAX_DROPOFF_DELAY = 2 * PICKUP_DELAY

status("Setup: {} agents, K={}".format(CALIB_NUM_AGENTS, CALIB_NUM_CLUSTERS))
envt = NYEnvironment(CALIB_NUM_AGENTS, START_EPOCH=0, STOP_EPOCH=24 * 3600,
                      MAX_CAPACITY=CAPACITY, EPOCH_LENGTH=DECISION_INTERVAL,
                      NUM_CLUSTERS=CALIB_NUM_CLUSTERS)
kmeans = KMeans(n_clusters=CALIB_NUM_CLUSTERS, random_state=0).fit(np.array(envt.travel_time))

inter_cluster_distance = np.zeros((CALIB_NUM_CLUSTERS, CALIB_NUM_CLUSTERS), dtype='float32')
centers = kmeans.cluster_centers_
dists = [np.linalg.norm(centers[i] - centers[j])
         for i in range(CALIB_NUM_CLUSTERS) for j in range(CALIB_NUM_CLUSTERS) if i != j]
maxd, mind = max(dists), min(dists)
for i in range(CALIB_NUM_CLUSTERS):
    for j in range(CALIB_NUM_CLUSTERS):
        inter_cluster_distance[i][j] = (np.linalg.norm(centers[i] - centers[j]) / maxd
                                         if i != j else 0.9 * mind / maxd)

envt.cluster_node_dict = kmeans
envt.e = 10
oracle = Oracle(envt)
central_agent = CentralAgent(envt)
value_function = PathBasedNN(envt, load_model_loc=None)
value_function.model.load_weights(
    '../models/NeurADP+SoftplusPathBasedNN_500agent_4capacity_90delay_60interval'
    '_vanilla_0sta_24end_2startday_11endday_8trained.h5')

ms.log = {}
random.seed(0)
pickup_avg = ms.run_epoch(envt, oracle, central_agent, kmeans, value_function,
                          VALIDATION_DAY, is_training=False,
                          inter_cluster_distance=inter_cluster_distance,
                          lamb=None, cont=True,
                          a_arr=[0.0] * 8, alpha_arr=[0.0] * 8,
                          downsample=DOWNSAMPLE)


def evaluate(a_val, alpha_val, label, a_arr_override=None):
    ms.log = {}
    ms.day = VALIDATION_DAY
    envt.recent_request_history.clear()
    random.seed(0)
    initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training=False)
    agents = [LearningAgent(i, s) for i, s in enumerate(initial_states)]
    a_arr = a_arr_override if a_arr_override is not None else [a_val] * 8
    served = ms.run_epoch(envt, oracle, central_agent, kmeans, value_function,
                           VALIDATION_DAY, is_training=False,
                           agents_predefined=agents,
                           inter_cluster_distance=inter_cluster_distance,
                           predicted_demand=[0], pickup_avg=pickup_avg,
                           a_arr=a_arr, alpha_arr=[alpha_val] * 8,
                           downsample=DOWNSAMPLE)
    rec = {"label": label, "a": a_arr, "alpha": alpha_val, "served": served}
    with open(RESULTS_LOG, 'a') as f:
        f.write(json.dumps(rec) + "\n")
    status("RESULT {}: a={} alpha={} served={}".format(label, a_arr, alpha_val, served))
    return served


# lambda=0 reference is already known from the Step 0 run (10585) but re-run it
# here so every point in the curve comes from one process/context.
CANDIDATES = [
    (0.0,  0.0, "lambda=0 (reference)"),
    (-0.3, 0.0, "lambda=-0.3"),
    (-0.1, 0.0, "lambda=-0.1"),
    (0.1,  0.0, "lambda=+0.1"),
    (0.3,  0.0, "lambda=+0.3"),
    (0.5,  0.0, "lambda=+0.5"),
    (1.0,  0.0, "lambda=+1.0 (paper's Team-Temp-1 anchor)"),
    (0.1,  3.0, "lambda=+0.1, alpha=3 (distance-weighted P)"),
]
if sweep_args.lambdas:
    CANDIDATES = [(float(x), 0.0, "lambda={:+g}".format(float(x)))
                  for x in sweep_args.lambdas.split(',')]

results = []
if sweep_args.profile:
    label = "profile={}".format(PROFILE)
    results.append((label, evaluate(None, 0.0, label, a_arr_override=PROFILE)))
else:
    for a_val, alpha_val, label in CANDIDATES:
        results.append((label, evaluate(a_val, alpha_val, label)))

status("=== SWEEP COMPLETE ===")
if not sweep_args.profile:
    base = results[0][1]
    for label, served in results:
        status("{}: served={} ({:+.2f}% vs lambda=0)".format(label, served, 100.0 * (served - base) / base))
