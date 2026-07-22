import tensorflow as tf
from tensorflow.python.keras.backend import exp

from enum import unique
from Environment import NYEnvironment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Oracle import Oracle
from NeurADPplus import PathBasedNN, RewardPlusDelay, NeuralNetworkBased
from Experience import Experience
from Request import Request
from sklearn.cluster import KMeans
import numpy as np

from typing import List

import pdb
from copy import deepcopy
from itertools import repeat
from multiprocessing.pool import Pool
import argparse
import time

import random
random.seed(0)

def run_epoch(envt,
              oracle,
              central_agent,
              kmeans,
              value_function,
              DAY,
              is_training,
              predicted_demand,
              agents_predefined=None,
              TRAINING_FREQUENCY: int=1):

    # cluster info 

    req_history = []
    reality_history = []
    expetataion_history = []
    cluster_ids = envt.cluster_node_dict.labels_
    num_clusters = envt.cluster_node_dict.n_clusters
    stored_values = []
    stored_rewards = []
    t = 0
    # INITIALISATIONS
    Experience.envt = envt

    # Initialising agents
    if agents_predefined is not None:
        agents = deepcopy(agents_predefined)
    else:
        initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training)
        agents = [LearningAgent(agent_idx, initial_state) for agent_idx, initial_state in enumerate(initial_states)]

    # time is set to 0, for current location

    # ITERATING OVER TIMESTEPS
    print("DAY: {}".format(DAY))
    rate = 1.0
    request_generator = envt.get_request_batch(DAY, rate)
    total_value_generated = 0
    num_total_requests = 0
    while True:
        # Get new requests
        try:
            current_requests = next(request_generator)
            print("Day : ", DAY, " Current time: {}".format(envt.current_time))
            print("Number of new requests: {}".format(len(current_requests)))
            # req_history.append(len(current_requests))
            # continue
        except StopIteration:
            # return req_history
            log['expectation'] = expetataion_history
            log['reality'] = reality_history
            break

        hor = 15
        if (t >= hor):
            print("Time : ", t-hor)
            reality = 0
            pow = 1.0
            for j in range(hor):
                reality += pow * stored_rewards[t-hor+j]
                pow *= 0.74
            print("Expectation : ", stored_values[t-hor], "Reality : ",reality)
            expetataion_history.append(stored_values[t-hor])
            reality_history.append(reality)
        
        # Get feasible actions
        feasible_actions_all_agents = oracle.get_feasible_actions(agents, current_requests)

        # Score feasible actions
        experience = Experience(deepcopy(agents), feasible_actions_all_agents, envt.current_time, len(current_requests) / envt.NUM_AGENTS, rate * predicted_demand[t] / envt.NUM_AGENTS)
        t += 1
        # Score all feasible actions using the V NN
        print("Requesting value")
        sta = time.perf_counter()
        scored_actions_all_agents = value_function.get_value([experience]) # Variable name changed
        print("Got values")

        # Choose actions for each agent (This solves the ILP)

        scored_final_actions = central_agent.choose_actions(scored_actions_all_agents, is_training=is_training, epoch_num=envt.num_days_trained)
        # scored_final_actions = central_agent.choose_actions(scored_actions_all_agents, is_training=False, epoch_num=envt.num_days_trained)
        fin = time.perf_counter()
        print("Forward time : ", fin - sta)

        value_next_state = []
        value_next_state.extend([score for _, score in scored_final_actions]) 
        print(sum(value_next_state))
        stored_values.append(sum(value_next_state))

        # Assign final actions to agents
        for agent_idx, (action, _) in enumerate(scored_final_actions):
            agents[agent_idx].path = deepcopy(action.new_path)

        # Calculate reward for selected actions
        rewards = []
        for action, _ in scored_final_actions:
            reward = envt.get_reward(action)
            rewards.append(reward)
            total_value_generated += reward
        print("Reward for epoch: {}".format(sum(rewards)))

        stored_rewards.append(sum(rewards))

        # Update
        if (is_training):
            # Update replay buffer
            value_function.remember(experience)
            # Update value function every TRAINING_FREQUENCY timesteps
            num_ups = 2
            if ((int(envt.current_time) / int(envt.EPOCH_LENGTH)) % TRAINING_FREQUENCY == TRAINING_FREQUENCY - 1):
                print("UPDATE")
                sta = time.perf_counter()
                value_function.update(central_agent, num_ups)
                fin = time.perf_counter()
                print("Backward ime : ", (fin - sta) / num_ups)

                # Diagnostics
                # for action, score in scored_actions_all_agents[0]:
                #     print("{}: {}, {}, {}".format(score, action.requests, action.new_path, action.new_path.total_delay))
                # print()
                # for idx, (action, score) in enumerate(scored_final_actions[:10]):
                #     print("{}: {}, {}, {}".format(score, action.requests, action.new_path, action.new_path.total_delay))

        # Sanity check
        for agent in agents:
            assert envt.has_valid_path(agent)

        # Writing statistics to logs
        if(is_training == False):
            log['total_day_{}_time_{}'.format(day, envt.current_time)] =  len(current_requests)
            log['served_day_{}_time_{}'.format(day, envt.current_time)] = sum(rewards)
        
        # Simulate the passing of time
        envt.simulate_motion(agents, current_requests)
        num_total_requests += len(current_requests)

        if(is_training == False):
            print("Percentage Served : ", 100 * total_value_generated / num_total_requests)


    # Printing statistics for current epoch
    print('Number of requests accepted: {}'.format(total_value_generated))
    print('Number of requests seen: {}'.format(num_total_requests))
    if(is_training == False):
        log['total_day_{}'.format(day)] = num_total_requests
        log['served_day_{}'.format(day)] = total_value_generated

    return total_value_generated


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--capacity', type=int, default=4)
    parser.add_argument('-n', '--numagents', type=int, default=500)
    parser.add_argument('-d', '--pickupdelay', type=int, default=90)
    parser.add_argument('-t', '--decisioninterval', type=int, default=60)
    parser.add_argument('-m', '--modellocation', type=str)
    parser.add_argument('-p', '--plot', type=int)
    parser.add_argument('-z', '--pretrained', type=int)
    args = parser.parse_args()

    Request.MAX_PICKUP_DELAY = args.pickupdelay
    Request.MAX_DROPOFF_DELAY = 2 * args.pickupdelay

    # Constants
    START_HOUR: int = 0
    END_HOUR: int = 24
    NUM_EPOCHS: int = 1
    TRAINING_DAYS = [4, 9]
    VALID_DAYS: List[int] = [2]
    TEST_DAYS: List[int] = [13]
    VALID_FREQ: int = 1
    SAVE_FREQ: int = VALID_FREQ
    LOG_DIR: str = '../logs/{}agent_{}capacity_{}delay_{}interval/'.format(args.numagents, args.capacity, args.pickupdelay, args.decisioninterval)
    # LOG_FILE: str = '../logs/{}agent_{}capacity_{}delay_{}interval_vanilla_{}sta_{}end_{}startday_{}endday_{}numtrained.npy'.format(args.numagents, args.capacity, args.pickupdelay, args.decisioninterval,START_HOUR, END_HOUR, TRAINING_DAYS[0], TRAINING_DAYS[-1], 2)
    log = {}

    pre_trained = args.pretrained

    print("Pre Trained : ", pre_trained)

    # Initialising components
    # TODO: Save start hour not start epoch
    num_clusters = 2
    envt = NYEnvironment(args.numagents, START_EPOCH=START_HOUR * 3600, STOP_EPOCH=END_HOUR * 3600, MAX_CAPACITY=args.capacity, EPOCH_LENGTH=args.decisioninterval, NUM_CLUSTERS=num_clusters)
    # TODO : Run K-Means on envt.travel_time and form the clusters
    # Pass this node, cluster dict to run_epoch
    travel_times = np.array(envt.travel_time)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(travel_times)
    envt.cluster_node_dict = kmeans
    oracle = Oracle(envt)
    central_agent = CentralAgent(envt)
    value_function = PathBasedNN(envt, log_dir=LOG_DIR, load_model_loc=args.modellocation)

    PREDICTED_DEMAND_FILE: str = '../logs/discounted_request_history.npy'
    try:
        predicted_log = np.load(PREDICTED_DEMAND_FILE, allow_pickle='TRUE').item()
        predicted_demand = predicted_log[0]
    except FileNotFoundError:
        print("WARNING: {} not found — using an empty placeholder dict instead of real predicted demand.".format(PREDICTED_DEMAND_FILE))
        from collections import defaultdict
        predicted_demand = defaultdict(int)

    if(pre_trained):
        print("Using Trained Model")
        # train_file = 'NeurADP+NoNoise_10_720_four{}_{}agent_{}capacity_{}delay_{}interval_vanilla_{}sta_{}end_{}startday_{}endday_{}trained.h5'.format(type(value_function).__name__, args.numagents, args.capacity, args.pickupdelay, args.decisioninterval, START_HOUR, END_HOUR, TRAINING_DAYS[0], TRAINING_DAYS[-1], 1)
        train_file = 'NeurADP+Softplus{}_{}agent_{}capacity_{}delay_{}interval_vanilla_{}sta_{}end_{}startday_{}endday_{}trained.h5'.format(type(value_function).__name__, args.numagents, args.capacity, args.pickupdelay, args.decisioninterval, 0, 24, TRAINING_DAYS[0], TRAINING_DAYS[-1], 2)
        # train_file = '0.6batched{}_{}agent_{}capacity_{}delay_{}interval_vanilla_{}sta_{}end_{}startday_{}endday_{}trained.h5'.format(type(value_function).__name__, args.numagents, args.capacity, args.pickupdelay, args.decisioninterval, START_HOUR, END_HOUR, 3, 3, 1)
        value_function.model.load_weights('../models/' + train_file)

        # for day in VALID_DAYS:
        #     total_requests_served = run_epoch(envt, oracle, central_agent, kmeans, value_function, day, is_training=False)
        #     print("\n(TEST) DAY: {}, Requests: {}\n\n".format(day, total_requests_served))
        #     # test_score += total_requests_served
        
        for day in TEST_DAYS:
            # Initialising agents
            initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training=False)
            agents = [LearningAgent(agent_idx, initial_state) for agent_idx, initial_state in enumerate(initial_states)]
            total_requests_served = run_epoch(envt, oracle, central_agent, kmeans, value_function, day, is_training=False, agents_predefined=agents, predicted_demand=predicted_demand)
            print("\n(TEST) DAY: {}, Requests: {}\n\n".format(day, total_requests_served))
            LOG_FILE: str = '../logs/' + train_file + 'scoreNeurADP+{}agent_{}capacity_{}delay_{}interval_{}test.npy'.format(args.numagents, args.capacity, args.pickupdelay, args.decisioninterval, day)
            np.save(LOG_FILE, log)
            log = {}
    else:
        max_test_score = 0
        total_req_history = {}
        for epoch_id in range(NUM_EPOCHS):
            for day in TRAINING_DAYS:
                total_requests_served = run_epoch(envt, oracle, central_agent, kmeans, value_function, day, is_training=True, predicted_demand=predicted_demand)
                print("\nDAY: {}, Requests: {}\n\n".format(day, total_requests_served))
                value_function.model.save_weights('../models/NeurADP+Softplus{}_{}agent_{}capacity_{}delay_{}interval_vanilla_{}sta_{}end_{}startday_{}endday_{}trained.h5'.format(type(value_function).__name__, args.numagents, args.capacity, args.pickupdelay, args.decisioninterval, START_HOUR, END_HOUR, TRAINING_DAYS[0], TRAINING_DAYS[-1], envt.num_days_trained+1))
                print("Saving right after training!")
                envt.num_days_trained += 1