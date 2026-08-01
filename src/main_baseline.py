import tensorflow as tf
from tensorflow.python.keras.backend import exp

from enum import unique
from Environment import NYEnvironment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Oracle import Oracle
from CEVD import ImmediateReward
from Experience import Experience
from Request import Request
from sklearn.cluster import KMeans
import numpy as np

from typing import List

from copy import deepcopy
import argparse
import time


def run_epoch(envt,
              oracle,
              central_agent,
              kmeans,
              value_function,
              DAY,
              is_training,
              agents_predefined=None):

    Experience.envt = envt

    if agents_predefined is not None:
        agents = deepcopy(agents_predefined)
    else:
        initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training)
        agents = [LearningAgent(agent_idx, initial_state) for agent_idx, initial_state in enumerate(initial_states)]

    print("DAY: {}".format(DAY))
    request_generator = envt.get_request_batch(DAY)
    total_value_generated = 0
    num_total_requests = 0
    while True:
        try:
            current_requests = next(request_generator)
            print("Day : ", DAY, " Current time: {}".format(envt.current_time))
            print("Number of new requests: {}".format(len(current_requests)))
        except StopIteration:
            log['expectation'] = []
            log['reality'] = []
            break

        feasible_actions_all_agents = oracle.get_feasible_actions(agents, current_requests)

        experience = Experience(deepcopy(agents), feasible_actions_all_agents, envt.current_time, len(current_requests))

        print("Requesting value")
        sta = time.perf_counter()
        scored_actions_all_agents = value_function.get_value([experience])
        print("Got values")

        scored_final_actions = central_agent.choose_actions(scored_actions_all_agents, is_training=is_training, epoch_num=envt.num_days_trained)
        fin = time.perf_counter()
        print("Forward time : ", fin - sta)

        for agent_idx, (action, _) in enumerate(scored_final_actions):
            agents[agent_idx].path = deepcopy(action.new_path)

        rewards = []
        for action, _ in scored_final_actions:
            reward = envt.get_reward(action)
            rewards.append(reward)
            total_value_generated += reward
        print("Reward for epoch: {}".format(sum(rewards)))

        for agent in agents:
            assert envt.has_valid_path(agent)

        if (is_training == False):
            log['total_day_{}_time_{}'.format(DAY, envt.current_time)] = len(current_requests)
            log['served_day_{}_time_{}'.format(DAY, envt.current_time)] = sum(rewards)

        envt.simulate_motion(agents, current_requests)
        num_total_requests += len(current_requests)

        if (is_training == False):
            print("Percentage Served : ", 100 * total_value_generated / num_total_requests)

    print('Number of requests accepted: {}'.format(total_value_generated))
    print('Number of requests seen: {}'.format(num_total_requests))
    if (is_training == False):
        log['total_day_{}'.format(DAY)] = num_total_requests
        log['served_day_{}'.format(DAY)] = total_value_generated

    return total_value_generated


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--capacity', type=int, default=4)
    parser.add_argument('-n', '--numagents', type=int, default=500)
    parser.add_argument('-d', '--pickupdelay', type=int, default=90)
    parser.add_argument('-t', '--decisioninterval', type=int, default=60)
    args = parser.parse_args()

    Request.MAX_PICKUP_DELAY = args.pickupdelay
    Request.MAX_DROPOFF_DELAY = 2 * args.pickupdelay

    START_HOUR: int = 0
    END_HOUR: int = 24
    TEST_DAYS_TO_RUN = [14, 15, 16, 17, 18]
    log = {}

    num_clusters = 2
    envt = NYEnvironment(args.numagents, START_EPOCH=START_HOUR * 3600, STOP_EPOCH=END_HOUR * 3600, MAX_CAPACITY=args.capacity, EPOCH_LENGTH=args.decisioninterval, NUM_CLUSTERS=num_clusters)
    travel_times = np.array(envt.travel_time)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(travel_times)
    envt.cluster_node_dict = kmeans
    oracle = Oracle(envt)
    central_agent = CentralAgent(envt)

    # Baseline: pure greedy immediate-reward matching, no training needed
    # (RewardPlusDelay with DELAY_COEFFICIENT=0 -- update()/remember() are no-ops).
    value_function = ImmediateReward()

    per_day_served = []
    per_day_seen = []
    for day in TEST_DAYS_TO_RUN:
        initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training=False)
        agents = [LearningAgent(agent_idx, initial_state) for agent_idx, initial_state in enumerate(initial_states)]
        total_requests_served = run_epoch(envt, oracle, central_agent, kmeans, value_function, day, is_training=False, agents_predefined=agents)
        print("\n(TEST) DAY: {}, Requests: {}\n\n".format(day, total_requests_served))
        per_day_served.append(total_requests_served)
        per_day_seen.append(log['total_day_{}'.format(day)])
        LOG_FILE: str = '../logs/BaselineImmediateReward_{}agent_{}capacity_{}delay_{}interval_{}test.npy'.format(args.numagents, args.capacity, args.pickupdelay, args.decisioninterval, day)
        np.save(LOG_FILE, log)
        log = {}

    per_day_served = np.array(per_day_served, dtype='float64')
    print("\nBaseline requests served per test day: {}".format(per_day_served.tolist()))
    print("Baseline requests seen per test day: {}".format(per_day_seen))
    print("Baseline mean requests served: {:.2f} +/- {:.2f}".format(per_day_served.mean(), per_day_served.std()))
