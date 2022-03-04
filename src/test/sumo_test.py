# simulation step
import logging
import subprocess
import sys

import traci

EPOCH = 500


def simulation_start():
    # simulation port and software
    PORT = 8813
    sumoBinary_gui = "/usr/share/sumo/bin/sumo-gui"
    sumoBinary = "/usr/share/sumo/bin/sumo"

    # config traffic light file
    rou_path = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/signal/TLS.rou.xml"
    auto_cfg_filepath = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/sumoFiles/signal/TLS.sumocfg"
    sumoProcess = subprocess.Popen([sumoBinary_gui, "-c", auto_cfg_filepath, "--remote-port", str(PORT), "--start"],
                                   stdout=sys.stdout, stderr=sys.stderr)
    logging.info("start SUMO GUI.")

    traci.init(PORT)
    logging.info("start TraCI.")

    return sumoProcess


for it in range(50):
    sumoProcess = simulation_start()
    for sim_step in range(EPOCH):
        traci.simulationStep()
        if sim_step > EPOCH - 10:
            traci.close()
            sumoProcess.kill()
            break