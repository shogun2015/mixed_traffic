import sys
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
    parser.add_argument("--control_interval", type=int, default=25, help="The control interval ")
    parser.add_argument("--gui", action="store_true", default=False, help="Visible")
    parser.add_argument("--port", type=int, default=8813, help="The port of gui")
    parser.add_argument("--probability", type=float, default=0.01, help="The total probability of IDV and HDV")
    parser.add_argument("--icv_ratio", type=float, default=0.00001, help="The ratio of icv/hdv")
    return parser.parse_args()


args = parse_args()

params["batch_size"] = args.batch_size
params["reload"] = args.reload
params["reload_exp"] = args.reload_exp
params["algorithm_name"] = args.alg_name
params["test"] = args.test
params["gamma"] = 0.99
params["control_interval"] = args.control_interval
params["gui"] = args.gui
params["port"] = args.port
params["probability"] = args.probability
params["icv_ratio"] = args.icv_ratio
params["exp_name"] = args.exp_name

date_str = time.strftime("%m-%d")
addInformation = ""
addInformation += "batch_%s" % params["batch_size"]
addInformation += "_"
addInformation += args.exp_name

params["directory"] = "output/{}/{}".format(date_str, addInformation)
controller = algorithm.make(params["algorithm_name"], params)

experiment_start = True

if params["reload"]:
    params['exp_name'] = params["reload_exp"].split("/")[-1]
    print("Test Exp Name:{}".format(params['exp_name']))
    params["reload_exp"] = join("output", params["reload_exp"])
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
