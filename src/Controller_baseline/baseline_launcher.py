import subprocess
import sys
import traci
import csv

import numpy as np

from controller_timer import controller_timer
from controller_greedy import controller_greedy
from alterXML import alterDemand
from util_test import util_test

sumoBinary_gui = "/usr/share/sumo/bin/sumo-gui"
sumoBinary = "/usr/share/sumo/bin/sumo"
auto_cfg_filepath = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/signal/TLS.sumocfg"
tls_rou_path = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/signal/TLS.rou.xml"

probability_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
ICV_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 1]

PORT = 10002
EPOCH = 10000


def simulation_start():
    sumoBinary_gui = "/usr/share/sumo/bin/sumo-gui"
    sumoBinary = "/usr/share/sumo/bin/sumo"
    # config traffic light file
    rou_path = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/signal/TLS.rou.xml"
    auto_cfg_filepath = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/signal/TLS.sumocfg"
    sumoProcess = subprocess.Popen([sumoBinary, "-c", auto_cfg_filepath, "--remote-port", str(PORT), "--start"],
                                   stdout=sys.stdout, stderr=sys.stderr)
    traci.init(PORT)
    return sumoProcess


if __name__ == '__main__':

    controller_timer = controller_timer()
    # controller_greedy = controller_greedy()
    test_env = util_test()

    mat_time = []
    mat_step = []
    mat_waiting_veh_lane = []
    for prob in probability_list:
        list_row_time = []
        list_row_step = []
        list_row_waiting_veh_lane = []
        for ratio in ICV_ratios:
            alterDemand(tls_rou_path, prob, ratio)
            repeat_row_time = []
            repeat_row_step = []
            repeat_row_waiting_veh_lane = []
            # repeat a certain times
            for index in range(3):
                sumoProcess = simulation_start()
                travel_time = {}
                sim_step = 0
                max_lane_waiting_veh_num_episode = 0
                for sim_step in range(EPOCH):
                    # SUMO step
                    traci.simulationStep()
                    # feature extraction
                    ICVs_control_list, travel_time, lane_static_veh_num = test_env.feature_step()

                    max_lane_waiting_veh_num = max(lane_static_veh_num)
                    if max_lane_waiting_veh_num > max_lane_waiting_veh_num_episode:
                        max_lane_waiting_veh_num_episode = max_lane_waiting_veh_num

                    min_lane_num = min(lane_static_veh_num)
                    max_lane_num = max(lane_static_veh_num)

                    if min_lane_num > 3 or max_lane_num > 20:
                        break

                    if sim_step % 25 == 0:
                        control_step = sim_step // 50
                        action_list = controller_timer.run_step(control_step)
                        # action_list = controller_greedy.run_step(lane_static_veh_num)
                        test_env.ICV_control(action_list)

                avg_travel_time = np.mean(list(travel_time.values()))
                print("scenario end: step:{}, travel_time:{}".format(sim_step, avg_travel_time))
                repeat_row_time.append(avg_travel_time)
                repeat_row_step.append(sim_step)
                repeat_row_waiting_veh_lane.append(max_lane_waiting_veh_num_episode)

                traci.close()
                sumoProcess.kill()

            list_row_time.append(np.mean(repeat_row_time))
            list_row_step.append(np.mean(repeat_row_step))
            list_row_waiting_veh_lane.append(np.mean(repeat_row_waiting_veh_lane))

        mat_time.append(list_row_time)
        mat_step.append(list_row_step)
        mat_waiting_veh_lane.append(list_row_waiting_veh_lane)

    f = open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/timer_time.csv", "w+", newline='')
    csv_writer = csv.writer(f)
    for row in mat_time:
        csv_writer.writerow(row)
    f.close()

    f = open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/timer_step.csv", "w+",
             newline='')
    csv_writer = csv.writer(f)
    for row in mat_step:
        csv_writer.writerow(row)
    f.close()

    f = open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/timer_waiting_lane.csv", "w+",
             newline='')
    csv_writer = csv.writer(f)
    for row in mat_waiting_veh_lane:
        csv_writer.writerow(row)
    f.close()

    # f = open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/greedy_time.csv", "w+",
    #          newline='')
    # csv_writer = csv.writer(f)
    # for row in mat_time:
    #     csv_writer.writerow(row)
    # f.close()
    #
    # f = open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/greedy_step.csv", "w+",
    #          newline='')
    # csv_writer = csv.writer(f)
    # for row in mat_step:
    #     csv_writer.writerow(row)
    # f.close()
    #
    # f = open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/greedy_waiting_lane.csv", "w+",
    #          newline='')
    # csv_writer = csv.writer(f)
    # for row in mat_waiting_veh_lane:
    #     csv_writer.writerow(row)
    # f.close()

    print("Done!!!")

