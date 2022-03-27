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
    parser = argparse.ArgumentParser("Experiments for multi-vehicle pursuit environments")
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
    parser.add_argument("--rou", type=int, default="", help="rou file sequence number")
    return parser.parse_args()


args = parse_args()


# Get an available Port for SUMO TraCI
def getPort():
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt = random.randint(15000, 20000)
    while tt in procarr:
        tt = random.randint(15000, 20000)
    return tt


params["batch_size"] = args.batch_size
params["reload"] = args.reload
params["reload_exp"] = args.reload_exp
params["algorithm_name"] = args.alg_name
params["test"] = args.test
params["gamma"] = 0.99
params["gui"] = args.gui
params["port"] = getPort()
# params["probability"] = args.probability
# params["icv_ratio"] = args.icv_ratio
params["exp_name"] = args.exp_name
params['rou'] = args.rou
params['action_gen'] = args.action_gen
params['control_interval'] = args.control_interval

print("SUMO TraCI Port: {}".format(params["port"]))

if params["test"]:
    params['exp_name'] = params["reload_exp"].split("/")[-2]
    params["directory"] = params['reload_exp']
    # print(params["reload_exp"])
    print("Test Exp Name:{}".format(params['exp_name']))
    # params["reload_exp"] = join("..", params["reload_exp"])
    if os.path.exists(params["reload_exp"]):
        spilt_exp_name = params['exp_name'].split("_")
        index_ControlInterval = spilt_exp_name.index("ControlInterval") + 1
        params["control_interval"] = int(spilt_exp_name[index_ControlInterval])
        index_ActionGen = spilt_exp_name.index("ActionGen") + 1
        params["action_gen"] = spilt_exp_name[index_ActionGen]

        print("Control Interval: {}".format(params["control_interval"]))
        print("Action Generation Method: " + params["action_gen"])

        print("Load model success")
    else:
        print("Not found model")
        experiment_start = False
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
