nr_steps = 2000000

params = {}
params["use_global_reward"] = True
params["save_summaries"] = True
params["save_test_summaries"] = True
params["alpha"] = 0.001
params["Episode"] = 10000
# params["input_shape"] = [,]
params["num_actions"] = 8
# params["num_actions"] = 256 # 2^8

# These hyperparameters are only required for DQN, VDN, QMIX
params["warmup_phase"] = 10000
params["target_update_period"] = 4000
params["memory_capacity"] = 20000
params["epsilon_decay"] = 1.0/50000


# format 0:10*5*5; 1:2*13*13; 2:3*13*13
# params["local_observation_format"] = 0

# Uncomment to manually set random seed
GLOBAL_SEED = 42
import torch
import numpy
import random
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(GLOBAL_SEED)
numpy.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)