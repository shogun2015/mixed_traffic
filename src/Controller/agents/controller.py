import random
import numpy
import torch
from utils import get_param_or_default
from utils import pad_or_truncate_sequences
from GAT.utils import *
from const import const_var
from torch.autograd import Variable
from action_generator import *


class ReplayMemory:

    def __init__(self, capacity, is_prioritized=False):
        self.transitions = []
        self.capacity = capacity
        self.nr_transitions = 0

    def save(self, transition):
        self.transitions.append(transition)
        self.nr_transitions += len(transition[0])
        if self.nr_transitions > self.capacity:
            removed_transition = self.transitions.pop(0)
            self.nr_transitions -= len(removed_transition[0])

    def sample_batch(self, minibatch_size):
        nr_episodes = self.size()
        if nr_episodes > minibatch_size:
            return random.sample(self.transitions, minibatch_size)
        return self.transitions

    def clear(self):
        self.transitions.clear()
        self.nr_transitions = 0

    def size(self):
        return len(self.transitions)


class Controller:

    def __init__(self, params):
        self.nr_actions = params["num_actions"]
        self.input_shape = [8, 2]
        self.actions = list(range(self.nr_actions))
        self.randomized_adversary_ratio = False
        self.gamma = params["gamma"]
        self.alpha = get_param_or_default(params, "alpha", 0.001)

    def policy(self, observations, adj, training_mode=True):
        random_joint_action = [random.choice(2) for _ in self.nr_actions]
        return random_joint_action

    def update(self, last_state, last_action, reward, state):
        return True


class DeepLearningController(Controller):

    def __init__(self, params):
        super(DeepLearningController, self).__init__(params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print(self.device)
        self.params = params
        self.use_global_reward = get_param_or_default(params, "use_global_reward", True)
        self.memory = ReplayMemory(params["memory_capacity"])
        self.warmup_phase = params["warmup_phase"]
        self.episode_transitions = []
        self.max_history_length = get_param_or_default(params, "max_history_length", 1)
        self.target_update_period = params["target_update_period"]
        self.epsilon = 1
        self.epsilon_decay = 1.0/200
        self.epsilon_min = 0.01
        self.training_count = 0
        self.current_histories = []
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.policy_net = None
        self.target_net = None
        self.summary = None
        self.loss_num = 0

        adj = const_var.lane_adjacent
        adj = normalize_adj(adj + np.eye(adj.shape[0]))
        adj = torch.FloatTensor(adj)
        if torch.cuda.is_available():
            adj = adj.cuda()
        self.adj = Variable(adj)

    def save_weights(self, path):
        if self.policy_net is not None:
            self.policy_net.save_weights(path)

    def load_weights(self, path):
        if self.policy_net is not None:
            self.policy_net.load_weights_from_history(path)

    def policy(self, observations, adj, training_mode=True):
        self.current_histories = observations
        action_probs = self.joint_action_probs(self.current_histories, adj, training_mode)
        action = np.copy(action_probs)

        # action[action > 0.5] = 1
        # action[action <= 0.5] = 0
        action = Q_to_Action(action)

        return action
        # return [numpy.random.choice(self.actions, p=probs) for probs in action_probs]

    def joint_action_probs(self, histories, adj, training_mode=True, agent_ids=None):
        return [numpy.ones(self.nr_actions) / 2 for _ in self.nr_actions]

    def update(self, last_state, last_action, reward, state):
        return self.update_transition(last_state, last_action, reward, state)

    def update_transition(self, last_state, last_action, reward, state):
        self.warmup_phase = max(0, self.warmup_phase - 1)
        pro_probs = self.joint_action_probs(last_state, self.adj, training_mode=True)
        self.memory.save((last_state, last_action, pro_probs, reward, state))
        return True

    def collect_minibatch_data(self, minibatch, whole_batch=False):
        last_states = []
        last_actions = []
        last_actions_prob = []
        rewards = []
        states = []
        for episode in minibatch:
            last_state, last_action, last_action_prob, reward, state = episode
            max_length = 1
            min_index = 0
            max_index = len(last_state)
            if whole_batch:
                indices = range(min_index, max_index)
            else:
                indices = [numpy.random.randint(min_index, max_index)]

            last_states.append(last_state)
            states.append(state)
            last_actions.append(last_action)
            last_actions_prob.append(last_action_prob)
            rewards.append(reward)
        return {"last_states": torch.tensor(last_states, device=self.device, dtype=torch.float32),
                "last_actions": torch.tensor(last_actions, device=self.device, dtype=torch.float32),
                "last_actions_prob": torch.tensor(last_actions_prob, device=self.device, dtype=torch.float32),
                "rewards": torch.tensor(rewards, device=self.device, dtype=torch.float32),
                "states": torch.tensor(states, device=self.device, dtype=torch.float32)}

    def normalized_returns(self, discounted_returns):
        R_mean = numpy.mean(discounted_returns)
        R_std = numpy.std(discounted_returns)
        return (discounted_returns - R_mean) / (R_std + self.eps)

    def update_target_network(self):
        target_net_available = self.target_net is not None
        if target_net_available and self.training_count % self.target_update_period is 0:
            self.target_net.protagonist_net.load_state_dict(self.policy_net.protagonist_net.state_dict())
            self.target_net.protagonist_net.eval()
