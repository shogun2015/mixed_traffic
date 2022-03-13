import logging
import subprocess
import sys

import traci

from alterXML import *

# simulation port and software
PORT = 10010
sumoBinary_w_gui = "/usr/share/sumo/bin/sumo-gui"
sumoBinary = "/usr/share/sumo/bin/sumo"
# simulation step
EPOCH = 7200

in_loop = ['ui_0', 'ui_1', 'di_0', 'di_1', 'ri_0', 'ri_1', 'li_0', 'li_1']
exit_loop = ['uo_0', 'uo_1', 'do_0', 'do_1', 'ro_0', 'ro_1', 'lo_0', 'lo_1']

# # config no-signal file
# rou_path = "../sumoFiles/no_signal/intersection.rou.xml"
# auto_cfg_filepath = '../sumoFiles/no_signal/intersection.sumocfg'
# # signal_cfg_filepath = '../sumoFiles/signal_intersection/junction.sumocfg'
# # result_file = '../results/demand_compare.csv'

# config traffic light file
rou_path = "../sumoFiles/signal/TLS.rou.xml"
auto_cfg_filepath = '../sumoFiles/signal/TLS.sumocfg'

travelTimeDict = dict()

class TravelTime(object):
    """
    Record Travel Time
    """

    def __init__(self):
        self.entry_time = 0
        self.exit_time = 0

    def add_entry_time(self, time):
        self.entry_time = time

    def add_exit_time(self, time):
        self.exit_time = time


def collectTraveTime(time_step):
    """
    Collect each vehicle traveling time from induction loop
    :return: travel time list for all vehicles
    """

    for in_loop_id in in_loop:
        # [(veh_id, veh_length, entry_time, exit_time, vType), ...]
        for vehData in traci.inductionloop.getVehicleData(in_loop_id):
            veh_id = vehData[0]
            entry_time = vehData[2]
            if entry_time != -1.0:
                travelTimeDict[veh_id] = TravelTime()
                travelTimeDict[veh_id].add_entry_time(entry_time)
                # print("timestep:{} \t vid:{} \t entry:{} ".format(self.time_step, veh_id, entry_time))
    for out_loop_id in exit_loop:
        # [(veh_id, veh_length, entry_time, exit_time, vType), ...]
        for vehData in traci.inductionloop.getVehicleData(out_loop_id):
            veh_id = vehData[0]
            exit_time = vehData[3]
            if exit_time != -1.0 and veh_id in list(travelTimeDict.keys()):
                travelTimeDict[veh_id].add_exit_time(exit_time)
                print("timestep:{} \t vid:{} \t travel_time:{}".format(time_step, veh_id,
                                                                       travelTimeDict[veh_id].exit_time -
                                                                       travelTimeDict[veh_id].entry_time))


def simulation_start():
    # auto_cfg_filepath = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/2way-single-intersection/single-intersection.sumocfg"
    auto_cfg_filepath = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/Trafficsignal/TLS.sumocfg"
    # sumoProcess = subprocess.Popen([sumoBinary, "-c", auto_cfg_filepath, "--remote-port", str(PORT), "--start"],
    #                                stdout=sys.stdout, stderr=sys.stderr)

    sumoProcess = subprocess.Popen([sumoBinary_w_gui, "-c", auto_cfg_filepath, "--remote-port", str(PORT), "--start"],
                                   stdout=sys.stdout, stderr=sys.stderr)

    logging.info("start SUMO GUI.")

    traci.init(PORT)
    logging.info("start TraCI.")

    return sumoProcess


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("simulation start...")

    # tls_rou_path = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/2way-single-intersection/single-intersection.rou.xml"
    tls_rou_path = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/Trafficsignal/TLS.rou.xml"
    # alterDemand(tls_rou_path, 0.01, 1)
    sumoProcess = simulation_start()

    for sim_step in range(EPOCH):
        traci.simulationStep()
        # collectTraveTime(sim_step)
        vids = traci.vehicle.getIDList()
        signal_id = traci.trafficlight.getIDList()
        duration = traci.trafficlight.getPhaseDuration(signal_id[0])
        state = traci.trafficlight.getRedYellowGreenState(signal_id[0])
        phases = traci.trafficlight.getAllProgramLogics(signal_id[0])[0].phases
        for vid in vids:
            speed = traci.vehicle.getSpeed(vid)
            maxSpeed = traci.vehicle.getMaxSpeed(vid)
            print("v:{},v_max:{}".format(speed,maxSpeed))

    traci.close()
    sumoProcess.kill()
