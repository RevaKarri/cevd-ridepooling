from os.path import isfile, isdir
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.python.keras.backend import exp
tf.compat.v1.disable_eager_execution()

from enum import unique
from Environment import NYEnvironment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Oracle import Oracle
from CEVD import PathBasedNN, RewardPlusDelay, NeuralNetworkBased
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
import pickle

def run_epoch(envt,
              oracle,
              central_agent,
              kmeans,
              value_function,
              DAY,
              is_training,
              agents_predefined=None,
              TRAINING_FREQUENCY: int=1,
              inter_cluster_distance=None,
              lamb=1,
              decay=0.8,
              cont=False,
              pickup_avg=None,
              a=None,
              b=None,
              alpha=None,
              predicted_demand=None):

    # cluster info 
    reality_history = []
    expetataion_history = []
    stored_values = []
    stored_rewards = []
    t = 0
    r = 0
    cluster_ids = envt.cluster_node_dict.labels_
    num_clusters = envt.cluster_node_dict.n_clusters
    P = np.zeros((num_clusters, num_clusters), dtype='float32')
    # pickup_history = np.zeros(num_clusters, dtype='float32')
    if cont:
        pickup_avg = np.zeros(num_clusters, dtype='float32')
    div = 0
    base = 5.0
    # INITIALISATIONS
    Experience.envt = envt

    # Initialising agents
    if agents_predefined is not None:
        agents = deepcopy(agents_predefined)
    else:
        initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training)
        agents = [LearningAgent(agent_idx, initial_state) for agent_idx, initial_state in enumerate(initial_states)]

    b_arr = [0, 0, 0, 0, 0, 0, 0, 0]
    # Estimated lambda, alpha values
    if(args.numagents == 500 and args.capacity == 4 and args.pickupdelay == 90):
        a_arr = [-0.65, -0.45, -0.55, -0.6, -0.55, -0.55, -0.6, -0.55]
        alpha_arr = [7.0, 8.0, 5.0, -10.0, 0.0, -10.0, 10.0, -10.0]
    elif(args.numagents == 500 and args.capacity == 5 and args.pickupdelay == 90):
        a_arr = [-0.65, -0.55, -0.5, -0.5, -0.55, -0.45, -0.5, -0.55]
        alpha_arr = [-4.0, 5.0, 0.0, 0.0, 5.0, 3.0, 5.0, 6.0]
    elif(args.numagents == 500 and args.capacity == 4 and args.pickupdelay == 120):
        a_arr = [-0.65, -0.55, -0.55, -0.6, -0.6, -0.55, -0.55, -0.55]
        alpha_arr = [-3.0, 0.0, 4.0, 3.0, -4.0, -7.0, 6.0, 0.0]
    elif(args.numagents == 500 and args.capacity == 4 and args.pickupdelay == 150):
        a_arr = [-0.6, -0.2, -0.5, -0.5, -0.5, -0.45, -0.55, -0.45]
        alpha_arr = [7.0, -7.0, 4.0, 6.0, 0.0, -4.0, 0.0, -5.0]
    

    print("DAY: {}".format(DAY))
    request_generator = envt.get_request_batch(DAY)
    total_value_generated = 0
    num_total_requests = 0
    while True:
        # Get new requests
        try:
            current_requests = next(request_generator)
            print("Day : ", DAY, " Current time: {}".format(envt.current_time))
            print("Number of new requests: {}".format(len(current_requests)))
        except StopIteration:
            if cont:
                return pickup_avg / 1440.0
            log['expectation'] = expetataion_history
            log['reality'] = reality_history
            break

        
        # Observations prepared

        cluster_info = [[] for i in range(num_clusters)]
        for i in range(len(agents)):
            cluster_info[cluster_ids[agents[i].position.next_location - 1]].append(i)

        count = 0 # check if agents is 0-indexed or 1-indexed
        # check if agents[i] and i are same
        new_order = []
        new_cluster_info = []
        for i in range(num_clusters):
            temp_info = []
            for j in cluster_info[i]:
                new_order.append(j)
                temp_info.append(count)
                count += 1
            new_cluster_info.append(temp_info)

        pickup_future = np.zeros(num_clusters, dtype='float32')
        for i in range(len(current_requests)):
            pickup_id = cluster_ids[current_requests[i].pickup]
            pickup_future[pickup_id] += 1

        if cont:
            for i in range(num_clusters):
                pickup_avg[i] += pickup_future[i]
            continue

        hor = 25
        if (t >= hor):
            print("Time : ", t-hor)
            reality = 0
            pow = 1.0
            for j in range(hor):
                reality += pow * stored_rewards[t-hor+j]
                pow *= 0.83
            print("Expectation : ", stored_values[t-hor], "Reality : ",reality)
            expetataion_history.append(stored_values[t-hor])
            reality_history.append(reality)
        
        
        
        if (t % 180 == 0):
            a = a_arr[r]
            b = b_arr[r]
            alpha = alpha_arr[r]
            # Provision to have cluster dependent lambda
            lamb = a * np.exp(-b * pickup_avg)
            print("Max : ",max(lamb))
            print("Min : ",min(lamb))
            r += 1
        
        for i in range(num_clusters):
            for j in range(num_clusters):
                P[i][j] = np.exp(alpha * inter_cluster_distance[i][j])
                
        print("Max P : ", np.max(P))
        print("Min P : ", np.min(P))


        # Get feasible actions
        feasible_actions_all_agents = oracle.get_feasible_actions(agents, current_requests)
        new_feasible_actions_all_agents = [feasible_actions_all_agents[i] for i in new_order]
        
        new_agents = [agents[i] for i in new_order]
        experience = Experience(deepcopy(new_agents), new_feasible_actions_all_agents, envt.current_time, len(current_requests) / envt.NUM_AGENTS, predicted_demand[0] / envt.NUM_AGENTS)
        
        # Score all feasible actions using the V NN
        print("Requesting value")
        sta = time.perf_counter()
        scored_actions_all_agents = value_function.get_value(experiences=[experience], P=P, cluster_info=new_cluster_info, lamb=lamb) # Variable name changed
        print("Got values")
        
        # Choose actions for each agent (This solves the ILP)

        scored_final_actions = central_agent.choose_actions(scored_actions_all_agents, is_training=is_training, epoch_num=envt.num_days_trained)
        fin = time.perf_counter()
        print("Forward time : ", fin - sta)

        value_next_state = []
        value_next_state.extend([score for _, score in scored_final_actions]) 
        stored_values.append(sum(value_next_state))

        # Assign final actions to agents
        for agent_idx, (action, _) in enumerate(scored_final_actions):
            agents[new_order[agent_idx]].path = deepcopy(action.new_path) # need inverse mapping here

        # Calculate reward for selected actions
        rewards = []
        for action, _ in scored_final_actions:
            reward = envt.get_reward(action)
            rewards.append(reward)
            total_value_generated += reward
        print("Reward for epoch: {}".format(sum(rewards)))

        stored_rewards.append(sum(rewards))

        t += 1
        # Update
        if (is_training):
            # Update replay buffer
            value_function.remember(experience, P, new_cluster_info)
            # Update value function every TRAINING_FREQUENCY timesteps
            if ((int(envt.current_time) / int(envt.EPOCH_LENGTH)) % TRAINING_FREQUENCY == TRAINING_FREQUENCY - 1):
                print("UPDATE")
                sta = time.perf_counter()
                value_function.update(central_agent, 3)
                fin = time.perf_counter()
                print("Backward ime : ", (fin - sta) / 3)
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

        # break

    # Printing statistics for current epoch
    print('Number of requests accepted: {}'.format(total_value_generated))
    print('Number of requests seen: {}'.format(num_total_requests))
    if(is_training == False):
        log['total_day_{}'.format(day)] = num_total_requests
        log['served_day_{}'.format(day)] = total_value_generated

    return total_value_generated


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--capacity', type=int, default=4)
    parser.add_argument('-n', '--numagents', type=int, default=500)
    parser.add_argument('-d', '--pickupdelay', type=int, default=90)
    parser.add_argument('-t', '--decisioninterval', type=int, default=60)
    parser.add_argument('-m', '--modellocation', type=str)
    parser.add_argument('-c', '--numbercluster', type=int, default=100)
    parser.add_argument('-e', '--samplingperagent', type=int, default=10)
    parser.add_argument('-p', '--plot', type=int, default = 20)
    parser.add_argument('-a', '--a', type=float, default=1.0)
    parser.add_argument('-b', '--b', type=float, default=1.0)
    parser.add_argument('-l', '--alpha', type=float, default=1.0)

    args = parser.parse_args()

    Request.MAX_PICKUP_DELAY = args.pickupdelay
    Request.MAX_DROPOFF_DELAY = 2 * args.pickupdelay

    # Constants
    START_HOUR: int = 0
    END_HOUR: int = 24
    NUM_EPOCHS: int = 1
    TRAINING_DAYS: List[int] = [4, 9]
    VALID_DAYS: List[int] = [2]
    TEST_DAYS: List[int] = [15]
    VALID_FREQ: int = 1#4
    SAVE_FREQ: int = VALID_FREQ
    

    # Initialising components
    # TODO: Save start hour not start epoch
    num_clusters = args.numbercluster
    e = args.samplingperagent
    
    num_trained_test = 1

    LOG_DIR: str = '../logs/{}agent_{}capacity_{}delay_{}interval/'.format(args.numagents, args.capacity, args.pickupdelay, args.decisioninterval)
    LOG_FILE: str = '../logs/PT{}agent_{}capacity_{}delay_{}interval_{}numclusters_{}e_{}sta_{}end_{}startday_{}endday_{}numtrained.npy'.format(args.numagents, args.capacity, args.pickupdelay, args.decisioninterval,num_clusters,e, START_HOUR, END_HOUR, TRAINING_DAYS[0], TRAINING_DAYS[-1], num_trained_test)
    log = {}
    kmeans_loc = '../models/{}numclusters.sav'.format(num_clusters)

    #PREDICTED_DEMAND_FILE: str = '../logs/discounted_request_history.npy'
    #predicted_log = np.load(PREDICTED_DEMAND_FILE, allow_pickle='TRUE').item()
    #predicted_demand = predicted_log[0]

    pre_trained = True

    envt = NYEnvironment(args.numagents, START_EPOCH=START_HOUR * 3600, STOP_EPOCH=END_HOUR * 3600, MAX_CAPACITY=args.capacity, EPOCH_LENGTH=args.decisioninterval, NUM_CLUSTERS=num_clusters)
    # TODO : Run K-Means on envt.travel_time and form the clusters
    # Pass this node, cluster dict to run_epoch

    if(isfile(kmeans_loc)):
        print("Using Saved K-Means")
        #kmeans = pickle.load(open(kmeans_loc, 'rb'))
    else:
        print("Running new K-Means")
        travel_times = np.array(envt.travel_time)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(travel_times)
        #pickle.dump(kmeans, open(kmeans_loc, 'wb'))
        print("Saved K-Means")
    
    # Create K means centre distances as a c * c matrix
    inter_cluster_distance = np.zeros((num_clusters, num_clusters), dtype='float32')
    all_centers = kmeans.cluster_centers_
    all_dist = []
    # eps = 1e-3
    for i in range(num_clusters):
        for j in range(num_clusters):
            if(i - j):
                all_dist.append(np.linalg.norm(all_centers[i] - all_centers[j]))

    max_dist = max(all_dist)
    min_dist = min(all_dist)
    for i in range(num_clusters):
        for j in range(num_clusters):
            if(i-j):
                inter_cluster_distance[i][j] = np.linalg.norm(all_centers[i] - all_centers[j]) / max_dist
            else:
                inter_cluster_distance[i][j] = 0.9 * min_dist / max_dist
 
    print("max/min : ", min(all_dist) / max(all_dist))

    envt.cluster_node_dict = kmeans
    envt.e = args.samplingperagent
    # cluster_ids[node_id] will give the cluster_id of that node
    oracle = Oracle(envt)
    central_agent = CentralAgent(envt)
    #value_function = PathBasedNN(envt, log_dir=LOG_DIR, load_model_loc=args.modellocation)
    value_function = PathBasedNN(envt,load_model_loc=None)


    if(pre_trained):
        print("Using Trained Model")
        # train_file = '0.6batched{}_{}agent_{}capacity_{}delay_{}interval_vanilla_{}sta_{}end_{}startday_{}endday_{}trained.h5'.format(type(value_function).__name__, args.numagents, args.capacity, args.pickupdelay, args.decisioninterval, START_HOUR, END_HOUR, TRAINING_DAYS[0], TRAINING_DAYS[-1], 1)
        train_file = 'NeurADP+Softplus{}_{}agent_{}capacity_{}delay_{}interval_vanilla_{}sta_{}end_{}startday_{}endday_{}trained.h5'.format(type(value_function).__name__, args.numagents, args.capacity, args.pickupdelay, args.decisioninterval, 0, 24, TRAINING_DAYS[0], TRAINING_DAYS[-1], 2)
        #value_function.model.load_weights('../models/' + train_file)
        # value_function.model.load_weights('../models/batched{}_{}agent_{}capacity_{}delay_{}interval_vanilla_{}sta_{}end_{}startday_{}endday_{}trained.h5'.format(type(value_function).__name__, 1000, args.capacity, 300, args.decisioninterval, 0, 24, TRAINING_DAYS[0], TRAINING_DAYS[-1], 1), by_name=True)
        # value_function.model.load_weights('../models/MADP{}_{}agent_{}capacity_{}delay_{}interval_{}numclusters_{}l_{}sta_{}end_{}startday_{}endday_{}trained.h5'.format(type(value_function).__name__, args.numagents, args.capacity, args.pickupdelay, args.decisioninterval, num_clusters, args.lamb, START_HOUR, END_HOUR, TRAINING_DAYS[0], TRAINING_DAYS[-1], 1))

        for day in range(1,21):
        
            pickup_avg = run_epoch(envt, oracle, central_agent, kmeans, value_function, day, is_training=False, inter_cluster_distance=inter_cluster_distance, lamb=None, cont=True)
            print(pickup_avg)
            initial_states = envt.get_initial_states(envt.NUM_AGENTS, is_training=False)
            # print(min(initial_states))
            agents = [LearningAgent(agent_idx, initial_state) for agent_idx, initial_state in enumerate(initial_states)]
            total_requests_served = run_epoch(envt, oracle, central_agent, kmeans, value_function, day, is_training=False, agents_predefined=agents, inter_cluster_distance=inter_cluster_distance, predicted_demand=[0], pickup_avg=pickup_avg, a=args.a, b=args.b, alpha=args.alpha)
            
            print("\n(TEST) DAY: {}, Requests: {}\n\n".format(day, total_requests_served))
            LOG_FILE: str = '../logs/' + train_file + 'ExpRealMNeurADP+{}agent_{}capacity_{}delay_{}interval_{}test.npy'.format(args.numagents, args.capacity, args.pickupdelay, args.decisioninterval, day)
            np.save(LOG_FILE, log)
            log = {}
                