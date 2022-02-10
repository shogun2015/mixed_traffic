import logging
import subprocess
import sys

import torch
import traci

from Controller.BasicController import BasicController
from GAT.models import GAT
from GAT.utils import *
from const import lane_adjacent

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

    # GAT-related
    nfeat = 5
    # feature:
    # 1 - Vehicle number in the corresponding junction lane
    # 2 - HDV number from the first ICV to stop line
    # 3 - Distance from the first ICV to stop line
    # 4 - Vehicle number after the first ICV
    # 5 - HDV number from the first ICV and the second ICV
    nclass = 2      # Two actions: enter / not enter junction
    # The specific below numbers get from pyGAT default number
    model = GAT(nfeat=nfeat, nhid=8, nclass=nclass, dropout=0.6, nheads=8, alpha=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    norm_lane_adj = normalize_adj(lane_adjacent)

    for sim_step in range(EPOCH):
        traci.simulationStep()
        # GAT-related process
        model.train()
        optimizer.zero_grad()
        features = controller.feature_step(timestep=sim_step)
        norm_feat = normalize_features(features)
        GAT_output = model(x=norm_feat, adj=norm_lane_adj)
        controller.run_step(GAT_output)

    traci.close()
    sumoProcess.kill()
