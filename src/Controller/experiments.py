import csv
import logging
import os
import random
import subprocess
import sys
import time
from os.path import join

import numpy
import numpy as np
import traci
from torch.utils.tensorboard import SummaryWriter

import data as data
from Controller.ICV_Controller import ICV_Controller
from GAT.utils import *
from const import const_var
from torch.autograd import Variable
from alterXML import *

probability_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
ICV_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 1]


def log(log_level, message_level, message):
    if message_level <= log_level:
        print(message)


def produce_choice_list():
    list = []
    for i in range(4):
        for j in range(10):
            list.append(i)
    return list


def simulation_start(params):
    # simulation port and software
    PORT = params["port"]
    if params["gui"]:
        sumoBinary = "/usr/share/sumo/bin/sumo-gui"
    else:
        sumoBinary = "/usr/share/sumo/bin/sumo"
    # config traffic light file
    rou_path = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/signal/TLS.rou.xml"
    auto_cfg_filepath = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/signal/TLS.sumocfg"
    sumoProcess = subprocess.Popen([sumoBinary, "-c", auto_cfg_filepath, "--remote-port", str(PORT), "--start"],
                                   stdout=sys.stdout, stderr=sys.stderr)
    logging.info("start SUMO GUI.")

    traci.init(PORT)
    logging.info("start TraCI.")

    return sumoProcess


def process_data(data, params, episode, type="test", summary=None):
    ThroughTime_ori = const_var.ThroughTime
    if not os.path.exists(os.path.join(params["directory"])):
        os.makedirs(os.path.join(params["directory"]))
    f = open(os.path.join(params["directory"], "result.csv"), "a+", newline='')
    csv_writer = csv.writer(f)
    # csv_writer.writerow(["Episode", "ThroughTimeDelta", "ThroughPassRatio"])
    vid_keys = data.keys()
    deltaTime = []
    passVheNums = 0
    for vid in vid_keys:
        times = data[vid]
        if times.exit_time != 0:
            lane = vid.split("_")[0]
            throughTimeDelta = times.exit_time - times.entry_time - ThroughTime_ori[lane]
            deltaTime.append(throughTimeDelta)
            passVheNums += 1
    totalVeh = len(vid_keys)
    if totalVeh == 0:
        print("0 total vehicle")
    csv_writer.writerow([episode, np.mean(deltaTime), passVheNums / totalVeh])
    if type == "train":
        summary.add_scalar('delatTime', np.mean(deltaTime), episode)
        summary.add_scalar('passRatio', passVheNums / totalVeh, episode)


def run(controller_rl, params, log_level=0):
    path = params["directory"]
    summary_write = SummaryWriter(path)
    controller_rl.summary = summary_write

    # simulation step
    EPOCH = 10000
    logging.basicConfig(level=logging.INFO)

    adj = const_var.lane_adjacent
    adj = normalize_adj(adj + np.eye(adj.shape[0]))
    adj = torch.FloatTensor(adj)
    if torch.cuda.is_available():
        adj = adj.cuda()
    adj = Variable(adj)

    tls_rou_path = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/signal/TLS.rou.xml"
    best_deltaTime = np.inf
    best_deltaNum = np.inf
    for episode in range(params["Episode"]):
        logging.info("The %s th episode simulation start..." % (episode))
        probability = np.random.choice(probability_list)
        icv_ratio = np.random.choice(ICV_ratios)
        alterDemand(tls_rou_path, probability, icv_ratio)
        sumoProcess = simulation_start(params)
        controller = ICV_Controller()
        # simulation environment related
        static_veh_num = 0
        static_veh_num_last_step = 0
        static_veh_in_junction_last = 0
        # last_avg_speed = 10
        last_state = None
        last_action = None
        summary_rewards = []
        action_exc = [0 for _ in range(8)]
        exit_vehicle_num_sum = 0
        for sim_step in range(EPOCH):
            traci.simulationStep()
            """
            feature - 特征矩阵
            static_veh_num - 静止车数目
            is_all_static_junction - 路口内是否全为静止车(弃用)
            min_lane_num - 静止车辆最多车道的车辆数
            max_lane_num - 静止车辆最少车道的车辆数
            exit_vehicle_num  - 离开车辆的数目
            """
            features, static_veh_num, lane_static_veh_num, exit_vehicle_num, travel_time = controller.feature_step(timestep=sim_step)
            exit_vehicle_num_sum += exit_vehicle_num

            min_lane_static_veh_num = min(lane_static_veh_num)
            max_lane_static_veh_num = max(lane_static_veh_num)


            if sim_step % params["control_interval"] == 0:
            # if sim_step % 25 == 0:

                # Reward:
                reward = static_veh_num_last_step - static_veh_num
                reward += 1 * (min_lane_static_veh_num - max_lane_static_veh_num)
                reward += -10 if min_lane_static_veh_num > 3 else 0
                reward += 1 * exit_vehicle_num_sum
                summary_rewards.append(reward)

                static_veh_num_last_step = static_veh_num
                exit_vehicle_num_sum = 0

                # Training the model
                if sim_step > 0:
                    policy_update = controller_rl.update(last_state, last_action, reward, features)

                # action = controller_rl.policy(norm_feat, controller_rl.adj, training_mode=True)
                action = controller_rl.policy(features, controller_rl.adj, training_mode=True)
                last_action = action
                action_exc = action
                # last_state = norm_feat
                last_state = features
                controller.run_step(action_exc)


            if sim_step > EPOCH - 10 or min_lane_static_veh_num > 3 or max_lane_static_veh_num > 20:
                # print("reset - avg_speed_junction: %s" % avg_speed_junction)
                # print(" sim_step：%s" % sim_step)
                # print(" min_lane_num：%s" % min_lane_num)
                # print(" max_lane_num：%s" % max_lane_num)
                process_data(controller.travelTimeDict, params, episode, type="train", summary=summary_write)
                traci.close()
                sumoProcess.kill()
                break

        # if episode % 36 == 0:
        #     deltaTime, deltaNum = test_suit(controller_rl, params)
        #     if deltaTime <= best_deltaTime and deltaNum <= best_deltaNum:
        #         best_deltaTime = deltaTime
        #         best_deltaNum = deltaNum
        #         print("Saving the best test model... The deltaTime : {} the deltaNume : {}".format(deltaTime, deltaNum))
        #         controller_rl.save_weights(path)
        #     else:
        #         if os.path.exists(os.path.join(path, "protagonist_model.pth")):
        #             print("Loading the best model...")
        #             controller_rl.load_weights(path)
        summary_write.add_scalar('Mean reward at each episode', numpy.mean(summary_rewards), episode)
        controller_rl.save_weights(path)
        print("The %s th episode's reward is: %s" % (episode, numpy.mean(summary_rewards)))
    log(log_level, 0, "DONE")
    return_values = {}
    # data.save_json(join(path, "returns.json"), return_values)
    # controller_rl.save_weights(path)
    return return_values


def test_suit(controller_rl, params):
    tls_rou_path = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/signal/TLS.rou.xml"
    EPOCH = 10000
    episode = 0
    mat_time = []
    mat_deltaNum = []
    for prob in probability_list:
    # for prob in [0.5]:
        list_row_time = []
        for ratio in ICV_ratios:
        # for ratio in [0.1]:
            episode += 1
            deltaNum = 0
            alterDemand(tls_rou_path, prob, ratio)
            logging.info("The %s th episode test simulation start..." % (episode))
            sumoProcess = simulation_start(params)
            controller = ICV_Controller()
            travel_time = {}
            max_lane_wait_veh_num_episode = 0
            # simulation environment related
            action_input = [0 for _ in range(8)]
            sim_step = 0
            for sim_step in range(EPOCH):
                traci.simulationStep()
                features, static_veh_num, lane_static_veh_num, exit_vehicle_num, travel_time \
                    = controller.feature_step(timestep=sim_step)

                max_lane_static_veh_num = max(lane_static_veh_num)
                min_lane_static_veh_num = min(lane_static_veh_num)

                if max_lane_static_veh_num > max_lane_wait_veh_num_episode:
                    max_lane_wait_veh_num_episode = max_lane_static_veh_num

                if max_lane_static_veh_num - min_lane_static_veh_num > deltaNum:
                    deltaNum = max_lane_static_veh_num - min_lane_static_veh_num

                if min_lane_static_veh_num > 3 or max_lane_static_veh_num > 20:
                    mat_deltaNum.append(deltaNum)
                    break

                if sim_step % params["control_interval"] == 0:
                    action = controller_rl.policy(features, controller_rl.adj, training_mode=True)
                    action_input = action
                    # print(action_input)
                controller.run_step(action_input)
            if len(travel_time.keys()) > 0:
                avg_travel_time = np.mean(list(travel_time.values()))
                print("Test suit {} : scenario end: step:{}, travel_time:{}".format(episode, sim_step, avg_travel_time))
                list_row_time.append(avg_travel_time)
            else:
                list_row_time.append(float('inf'))
                print("Test suit {} : scenario end: step:{}, travel_time: inf".format(episode, sim_step))

            traci.close()
            sumoProcess.kill()

        mat_time.append(list_row_time)

    return np.mean(mat_time), np.mean(mat_deltaNum)


def test(controller_rl, params, log_level=0):
    path = params["directory"]
    # simulation step
    EPOCH = 10000

    logging.basicConfig(level=logging.INFO)

    adj = const_var.lane_adjacent
    adj = normalize_adj(adj + np.eye(adj.shape[0]))
    adj = torch.FloatTensor(adj)
    if torch.cuda.is_available():
        adj = adj.cuda()
    adj = Variable(adj)

    tls_rou_path = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/signal/TLS.rou.xml"

    episode = 0
    mat_time = []
    mat_step = []
    mat_wait_veh_lane = []
    for prob in probability_list:
        list_row_time = []
        list_row_step = []
        list_row_wait_veh_lane = []
        for ratio in ICV_ratios:
            episode += 1
            alterDemand(tls_rou_path, prob, ratio)
            logging.info("The %s th episode simulation start..." % (episode))

            repeat_row_time = []
            repeat_row_step = []
            repeat_row_waiting_veh_lane = []
            for index in range(3):
                sumoProcess = simulation_start(params)
                controller = ICV_Controller()
                travel_time = {}
                max_lane_wait_veh_num_episode = 0
                # simulation environment related
                action_init = [0 for _ in range(8)]
                sim_step = 0
                for sim_step in range(EPOCH):
                    traci.simulationStep()
                    features, static_veh_num, lane_static_veh_num, exit_vehicle_num, travel_time \
                        = controller.feature_step(timestep=sim_step)

                    max_lane_static_veh_num = max(lane_static_veh_num)
                    min_lane_static_veh_num = min(lane_static_veh_num)

                    if max_lane_static_veh_num > max_lane_wait_veh_num_episode:
                        max_lane_wait_veh_num_episode = max_lane_static_veh_num

                    if min_lane_static_veh_num > 3 or max_lane_static_veh_num > 20:
                        break

                    if sim_step % params["control_interval"] == 0:
                        if params["reload"]:
                            action = controller_rl.policy(features, controller_rl.adj, training_mode=True)
                            action_init = action
                            controller.run_step(action)

                    # input_action = action_init
                    # controller.run_step(input_action)
                    # print(input_action)
                avg_travel_time = np.mean(list(travel_time.values()))
                # print("scenario end: step:{}, travel_time:{}".format(sim_step, avg_travel_time))
                repeat_row_time.append(avg_travel_time)
                repeat_row_step.append(sim_step)
                repeat_row_waiting_veh_lane.append(max_lane_wait_veh_num_episode)

                traci.close()
                sumoProcess.kill()

            list_row_time.append(np.mean(repeat_row_time))
            list_row_step.append(np.mean(repeat_row_step))
            list_row_wait_veh_lane.append(np.mean(repeat_row_waiting_veh_lane))

        mat_time.append(list_row_time)
        mat_step.append(list_row_step)
        mat_wait_veh_lane.append(list_row_wait_veh_lane)

    exp_name = params['reload_exp']
    path = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/"
    path_time = path + "{}_GAT_time.csv".format(params['exp_name'])
    path_step = path + "{}_GAT_step.csv".format(params['exp_name'])
    path_wait_lane = path + "{}_GAT_wait_lane.csv".format(params['exp_name'])

    f = open(path_time, "w+", newline='')
    csv_writer = csv.writer(f)
    for row in mat_time:
        csv_writer.writerow(row)
    f.close()

    f = open(path_step, "w+", newline='')
    csv_writer = csv.writer(f)
    for row in mat_step:
        csv_writer.writerow(row)
    f.close()

    f = open(path_wait_lane, "w+", newline='')
    csv_writer = csv.writer(f)
    for row in mat_wait_veh_lane:
        csv_writer.writerow(row)
    f.close()

    print("Test Done!!!")

    log(log_level, 0, "DONE")
    return_values = {}
    # data.save_json(join(path, "returns.json"), return_values)
    # controller_rl.save_weights(path)
    return return_values
