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

probability_list = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
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
    notpassVheNums = 0
    for vid in vid_keys:
        times = data[vid]
        if times.exit_time != 0:
            lane = vid.split("_")[0]
            throughTimeDelta = times.exit_time - times.entry_time - ThroughTime_ori[lane]
            deltaTime.append(throughTimeDelta)
        else:
            notpassVheNums += 1
    totalVeh = len(vid_keys)
    csv_writer.writerow([episode, np.mean(deltaTime), notpassVheNums / totalVeh])
    if type == "train":
        summary.add_scalar('delatTime', np.mean(deltaTime), episode)
        summary.add_scalar('passRatio', notpassVheNums / totalVeh, episode)


def run(controller_rl, params, log_level=0):
    path = params["directory"]
    summary_write = SummaryWriter(path)
    controller_rl.summary = summary_write

    # simulation step
    EPOCH = 7200
    logging.basicConfig(level=logging.INFO)

    adj = const_var.lane_adjacent
    adj = normalize_adj(adj + np.eye(adj.shape[0]))
    adj = torch.FloatTensor(adj)
    if torch.cuda.is_available():
        adj = adj.cuda()
    adj = Variable(adj)

    tls_rou_path = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/signal/TLS.rou.xml"
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
            is_all_static_junction - 路口内是否全为静止车
            min_lane_num - 静止车辆最多车道的车辆数
            max_lane_num - 静止车辆最少车道的车辆数
            exit_vehicle_num  - 离开车辆的数目
            """
            features, static_veh_num, is_all_static_junction, min_lane_num, max_lane_num, exit_vehicle_num \
                = controller.feature_step(timestep=sim_step)
            exit_vehicle_num_sum += exit_vehicle_num
            # print("exit sum:{}, eixt:{}".format(exit_vehicle_num_sum, exit_vehicle_num))
            if sim_step % params["control_interval"] == 0:
                # print("control step")

                # Reward:
                reward = static_veh_num_last_step - static_veh_num
                reward += -10 if is_all_static_junction is True else 0
                reward += min_lane_num - max_lane_num
                reward += exit_vehicle_num_sum
                # reward += (5 - max_lane_num) / 25
                summary_rewards.append(reward)

                static_veh_num_last_step = static_veh_num
                exit_vehicle_num_sum = 0

                # Training the model
                if sim_step > 0:
                    # policy_update = controller_rl.update(last_state, last_action, reward, norm_feat)
                    policy_update = controller_rl.update(last_state, last_action, reward, features)

                # If average speed of vehicles in junction is too slow, the junction is deadlock.
                # The simulation need to be reset
                # if (avg_speed_junction < 0.1 and last_avg_speed < 0.1) \
                #         or sim_step > EPOCH - 100 \
                #         or min_lane_num >= 5 \
                #         or max_lane_num >= 20:
                if sim_step > EPOCH - 10 or is_all_static_junction or min_lane_num > 5 or max_lane_num > 20:
                    # print("reset - avg_speed_junction: %s" % avg_speed_junction)
                    # print(" sim_step：%s" % sim_step)
                    # print(" min_lane_num：%s" % min_lane_num)
                    # print(" max_lane_num：%s" % max_lane_num)
                    traci.close()
                    sumoProcess.kill()
                    data = controller.travelTimeDict
                    process_data(data, params, episode, type="train", summary=summary_write)
                    break
                # action = controller_rl.policy(norm_feat, controller_rl.adj, training_mode=True)
                action = controller_rl.policy(features, controller_rl.adj, training_mode=True)
                last_action = action
                action_exc = action
                # last_state = norm_feat
                last_state = features
            controller.run_step(action_exc)

        summary_write.add_scalar('Mean reward at each episode', numpy.mean(summary_rewards), episode)
        print("The %s th episode's reward is: %s" % (episode, numpy.mean(summary_rewards)))

        # summary_write.add_scalar('protagonist_discounted_returns', protagonist_discounted_return, episode_id)
        # summary_write.add_scalar('protagonist_undiscounted_returns', protagonist_undiscounted_return, episode_id)
        # summary_write.add_scalar('training_discounted_returns', env.discounted_return, episode_id)
        # summary_write.add_scalar('training_undiscounted_returns', env.undiscounted_return, episode_id)
        # summary_write.add_scalar('training_domain_statistic', env.domain_statistic(), episode_id)
    log(log_level, 0, "DONE")
    return_values = {}
    data.save_json(join(path, "returns.json"), return_values)
    controller_rl.save_weights(path)
    return return_values


def test(controller_rl, params, log_level=0):
    path = params["directory"]
    # simulation step
    EPOCH = 7200

    logging.basicConfig(level=logging.INFO)

    adj = const_var.lane_adjacent
    adj = normalize_adj(adj + np.eye(adj.shape[0]))
    adj = torch.FloatTensor(adj)
    if torch.cuda.is_available():
        adj = adj.cuda()
    adj = Variable(adj)

    tls_rou_path = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/signal/TLS.rou.xml"
    # probability = np.random.choice(probability_list)
    # icv_ratio = np.random.choice(ICV_ratios)
    testNum = len(probability_list) * len(ICV_ratios)

    for episode in range(testNum):
        # alterDemand(tls_rou_path, probability_list[int(episode/len(probability_list))], ICV_ratios[episode % len(ICV_ratios)])
        alterDemand(tls_rou_path, 0.05, 0.1)
        logging.info("The %s th episode simulation start..." % (episode))
        sumoProcess = simulation_start(params)
        controller = ICV_Controller()
        # simulation environment related
        # action_init = [0 for _ in range(8)]
        for sim_step in range(EPOCH):
            traci.simulationStep()
            # print(sim_step)
            """
            feature - 特征矩阵
            static_veh_num - 静止车数目
            is_all_static_junction - 路口内是否全为静止车
            min_lane_num - 静止车辆最多车道的车辆数
            max_lane_num - 静止车辆最少车道的车辆数
            exit_vehicle_num  - 离开车辆的数目
            """
            features, static_veh_num, is_all_static_junction, min_lane_num, max_lane_num, exit_vehicle_num \
                = controller.feature_step(timestep=sim_step)
            # exit_vehicle_num_sum += exit_vehicle_num
            if sim_step % 50 == 0:
                if params["reload"]:
                    action = controller_rl.policy(features, controller_rl.adj, training_mode=True)
                    action_init = action
                    controller.run_step(action)

            # input_action = action_init
            # controller.run_step(input_action)
            # print(input_action)
    log(log_level, 0, "DONE")
    return_values = {}
    # data.save_json(join(path, "returns.json"), return_values)
    # controller_rl.save_weights(path)
    return return_values
