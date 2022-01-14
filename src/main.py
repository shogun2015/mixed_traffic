from Controller.BasicController import BasicController

import subprocess
import logging
import numpy as np
import pandas as pd
import traci
import sys

# simulation port and software
PORT = 8813
sumoBinary = "/usr/share/sumo/bin/sumo-gui"
# simulation step
EPOCH = 7200

# # config no-signal file
# rou_path = "../sumoFiles/no_signal/intersection.rou.xml"
# auto_cfg_filepath = '../sumoFiles/no_signal/intersection.sumocfg'
# # signal_cfg_filepath = '../sumoFiles/signal_intersection/junction.sumocfg'
# # result_file = '../results/demand_compare.csv'

# config traffic light file
rou_path = "../sumoFiles/signal/TLS.rou.xml"
auto_cfg_filepath = '../sumoFiles/signal/TLS.sumocfg'


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("simulation start...")

    sumoProcess = subprocess.Popen([sumoBinary, "-c", auto_cfg_filepath, "--remote-port", str(PORT)],
                                   stdout=sys.stdout, stderr=sys.stderr)
    logging.info("start SUMO GUI.")
    traci.init(PORT)
    logging.info("start TraCI.")

    controller = BasicController()
    logging.info("start " + controller.__str__() + "...")

    for sim_step in range(EPOCH):
        traci.simulationStep()
        controller.simulation_step()

    traci.close()
    sumoProcess.kill()
