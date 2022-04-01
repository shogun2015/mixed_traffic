import random
import sys

import const

sys.path.append("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/src/Controller")
import algorithm as algorithm
import experiments as experiments
import os
from os.path import join
from settings import params
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser("Experiments for mixed-traffic unsignalized intersectioni environments")
    parser.add_argument("--exp_name", type=str, default="test", help="adition name of the experiment")  # 实验名
    parser.add_argument("--batch_size", type=int, default=32, help="The train batch size")
    parser.add_argument("--alg_name", type=str, default="PPO", help="The algorithm name of training")
    parser.add_argument("--reload", action="store_true", default=False, help="reload the model")
    parser.add_argument("--reload_exp", type=str, default=None, help="The reload exp name")
    parser.add_argument("--test", action="store_true", default=False, help="Test")
    parser.add_argument("--control_interval", type=int, default=0, help="The control interval ")
    parser.add_argument("--gui", action="store_true", default=False, help="Visible")
    # parser.add_argument("--port", type=int, default=8813, help="The port of gui")
    # parser.add_argument("--probability", type=float, default=0.01, help="The total probability of IDV and HDV")
    # parser.add_argument("--icv_ratio", type=float, default=0.00001, help="The ratio of icv/hdv")
    parser.add_argument("--action_gen", type=str, default="", help="The action generation method")
    parser.add_argument("--use_gat", action="store_true", default=False, help="If use gat")
    parser.add_argument("--flow_type", type=str, default=" ", help="The vehicle flow type. [uniform, random, mixed]")
    parser.add_argument("--rou", type=int, default="", help="rou file sequence number")
    parser.add_argument("--add_loss", action="store_true", default=False, help="The Loss computer")
    return parser.parse_args()


args = parse_args()


# Get an available Port for SUMO TraCI
def getPort():
    pscmd = "netstat -atun |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    while True:
        port_candidate = random.randint(15000, 20000)
        if str(port_candidate) not in procarr:
            return port_candidate


params["batch_size"] = args.batch_size
params["reload"] = args.reload
params["reload_exp"] = args.reload_exp
params["algorithm_name"] = args.alg_name
params["test"] = args.test
params["gamma"] = 0.99
params["gui"] = args.gui
# params["port"] = args.port
params["port"] = getPort()
# params["probability"] = args.probability
# params["icv_ratio"] = args.icv_ratio
params["exp_name"] = args.exp_name
params['action_gen'] = args.action_gen
params['control_interval'] = args.control_interval
params['use_gat'] = args.use_gat
params['rou'] = args.rou
params['flow_type'] = args.flow_type
params['add_loss'] = args.add_loss

print("SUMO TraCI Port: {}".format(params["port"]))

if params['flow_type'] not in const.const_var.flows:
    print("Please specify the vehicle flow type: [uniform, random, mixed]")
    sys.exit()

if params["test"]:
    params['exp_name'] = params["reload_exp"].split("/")[-2]
    params["directory"] = params['reload_exp']
    # print(params["reload_exp"])
    print("Test Exp Name:{}".format(params['exp_name']))
    # params["reload_exp"] = join("..", params["reload_exp"])
    if os.path.exists(params["reload_exp"]):
        split_exp_name = params['exp_name'].split("_")
        index_alg = split_exp_name.index("alg") + 1
        params["algorithm_name"] = split_exp_name[index_alg]
        index_ControlInterval = split_exp_name.index("ControlInterval") + 1
        params["control_interval"] = int(split_exp_name[index_ControlInterval])
        index_ActionGen = split_exp_name.index("ActionGen") + 1
        params["action_gen"] = split_exp_name[index_ActionGen]
        if 'gat' in split_exp_name:
            params['use_gat'] = True
        if 'mlp' in split_exp_name:
            params['use_gat'] = False

        print("Control Interval: {}".format(params["control_interval"]))
        print("Action Generation Method: " + params["action_gen"])
        print("Feature Extraction Layer - GAT? {}".format(params['use_gat']))

        print("Load model success")
    else:
        print("Not found model")
else:
    if args.control_interval > 0:
        params["control_interval"] = args.control_interval
        print("Control Interval: {}".format(params["control_interval"]))
    else:
        print("Please specify control interval!!!")
        sys.exit()

    if args.action_gen in const.const_var.action_gen_list:
        params["action_gen"] = args.action_gen  # "thresh" or "QtoA"
        print("Action Generation Method: " + params["action_gen"])
    else:
        print("Please specify action generation method!!!")
        sys.exit()
    date_str = time.strftime("%m-%d")
    addInformation = ""
    # addInformation += "batch_%s" % params["batch_size"]
    # addInformation += "_"
    addInformation += args.exp_name
    addInformation += "_alg_" + params["algorithm_name"]
    addInformation += "_ControlInterval_{}".format(params["control_interval"])
    addInformation += "_ActionGen_{}".format(params["action_gen"])
    addInformation += "_gat" if params['use_gat'] else "_mlp"
    addInformation += "_" + params['flow_type']
    params["directory"] = "output/{}/{}".format(date_str, addInformation)


controller = algorithm.make(params["algorithm_name"], params)

experiment_start = True

if params["test"]:
    if os.path.exists(params["reload_exp"]):
        controller.load_weights(params["reload_exp"])
        print("Load model success")
    else:
        print("Not found model")
        experiment_start = False

if experiment_start:
    if args.test:
        result = experiments.test(controller, params, log_level=0)
    else:
        result = experiments.run(controller, params, log_level=0)
