import subprocess

import numpy as np
import pandas as pd
import traci
import sys

# simulation port and software
PORT = 8813
sumoBinary = "/usr/share/sumo/bin/sumo-gui"
# simulation step
EPOCH = 7200

# config file
rou_path = "../sumoFiles/no_signal/intersection.rou.xml"
auto_cfg_filepath = '../sumoFiles/no_signal/intersection.sumocfg'
# signal_cfg_filepath = '../sumoFiles/signal_intersection/junction.sumocfg'
# result_file = '../results/demand_compare.csv'


if __name__ == "__main__":
    print("simulation start...\n")

    sumoProcess = subprocess.Popen([sumoBinary, "-c", auto_cfg_filepath, "--remote-port", str(PORT)],
                                   stdout=sys.stdout, stderr=sys.stderr)
    traci.init(PORT)

    for sim_step in range(EPOCH):
        traci.simulationStep()



    traci.close()
    sumoProcess.kill()
