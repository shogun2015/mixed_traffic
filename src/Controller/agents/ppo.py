import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from agents.controller import DeepLearningController, get_param_or_default
from modules import MLP, AdversarialModule, MLP3D, TimeTransformer
import time
from GAT.models import GAT

class PPONet(nn.Module):
    def __init__(self, input_shape, nr_actions, max_history_length, q_values=False):
        super(PPONet, self).__init__()
        # self.fc_net = MLP(input_shape, max_history_length)
        self.fc_net = GAT(nfeat=84, nhid=8, nclass=2, dropout=0.6, nheads=8, alpha=0.2)
        self.forward_net = MLP(input_shape, max_history_length)
        # self.action_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions)
        self.action_head = nn.Linear(64, nr_actions)
        if q_values:
            # self.value_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions)
            self.value_head = nn.Linear(64, nr_actions)
        else:
            # self.value_head = nn.Linear(self.fc_net.nr_hidden_units, 1)
            self.value_head = nn.Linear(64, 1)
        self.action_head_sigmoid = nn.Sigmoid()

    def forward(self, x, adj, use_gumbel_softmax=False):
        x_gat = self.fc_net(x, adj)
        x = self.forward_net(x_gat)
        action_out = self.action_head(x)
        if use_gumbel_softmax:
            return F.gumbel_softmax(self.action_head(x), hard=True, dim=-1), self.value_head(x)

        # return F.softmax(action_out, dim=-1), self.value_head(x)
        return  self.action_head_sigmoid(action_out), self.value_head(x)


class PPOLearner(DeepLearningController):

    def __init__(self, params):
        super(PPOLearner, self).__init__(params)
        self.nr_epochs = get_param_or_default(params, "nr_epochs", 5)
        self.nr_episodes = get_param_or_default(params, "nr_episodes", 10)
        self.eps_clipping = get_param_or_default(params, "eps_clipping", 0.2)
        self.use_q_values = get_param_or_default(params, "use_q_values", False)
        self.warmup_phase_epochs = 50
        history_length = self.max_history_length
        input_shape = self.input_shape
        nr_actions = self.nr_actions
        network_constructor = lambda in_shape, actions, length: PPONet(in_shape, actions, length, self.use_q_values)
        self.policy_net = AdversarialModule(input_shape, nr_actions, history_length, network_constructor).to(self.device)
        self.protagonist_optimizer = torch.optim.Adam(self.policy_net.protagonist_parameters(), lr=self.alpha)

    def joint_action_probs(self, histories, adj, training_mode=True, agent_ids=None):
        history = torch.tensor(histories, device=self.device, dtype=torch.float32)
        # in_time = time.time()
        probs, value = self.policy_net(history, adj)
        len_probs = len(probs)
        # print("1Once time is %s" % (time.time() - in_time))
        # assert len(probs) == 1, "Expected length 1, but got shape {}".format(probs.shape)
        action_probs = probs.detach().cpu().numpy()[0]
        value = value.detach()
        if numpy.random.rand() <= self.epsilon:
            probs = numpy.random.random(action_probs.size)
            action_probs = probs

        return action_probs

    def policy_update(self, minibatch_data, optimizer, random_agent_indices=None):
        last_states = minibatch_data["last_states"]
        last_actions = minibatch_data["last_actions"]
        last_actions_prob = minibatch_data["last_actions_prob"]
        rewards = minibatch_data["rewards"]
        states = minibatch_data["states"]
        action_probs, expected_values = self.policy_net(last_states, self.adj)
        policy_losses = []
        value_losses = []
        for probs, action, value, R, old_prob in zip(action_probs, last_actions, expected_values, rewards, last_actions_prob):
            value_index = 0
            if self.use_q_values:
                value_index = action
                advantage = value[value_index].detach()
            else:
                advantage = R - value.item()
            policy_losses.append(self.policy_loss(advantage, probs, action, old_prob))
            value_losses.append(F.mse_loss(value[value_index], torch.tensor(R)))
        value_loss = torch.stack(value_losses).mean()
        policy_loss = torch.stack(policy_losses).mean()
        loss = policy_loss + value_loss
        if not self.params["test"]:
            self.summary.add_scalar("loss", loss, self.loss_num)
        self.loss_num += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return True

    def policy_loss(self, advantage, probs, action, old_prob):
        m1 = Categorical(probs)
        m2 = Categorical(old_prob)
        ratio = torch.exp(m1.log_prob(action) - m2.log_prob(action))
        clipped_ratio = torch.clamp(ratio, 1-self.eps_clipping, 1+self.eps_clipping)
        surrogate_loss1 = ratio*advantage
        surrogate_loss2 = clipped_ratio*advantage
        return -torch.min(surrogate_loss1, surrogate_loss2)

    def value_update(self, minibatch_data):
        pass

    def update(self, last_state, last_action, reward, state):
        super(PPOLearner, self).update(last_state, last_action, reward, state)
        if self.memory.size() >= self.nr_episodes:
            trainable_setting = True
            if trainable_setting:
                batch = self.memory.sample_batch(self.memory.capacity)
                minibatch_data = self.collect_minibatch_data(batch, whole_batch=True)
                self.value_update(minibatch_data)
                for _ in range(self.nr_epochs):
                    optimizer = self.protagonist_optimizer
                    self.policy_update(minibatch_data, optimizer)
                if self.warmup_phase_epochs <= 0:
                    self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
                self.warmup_phase_epochs -= 1
                self.warmup_phase_epochs = max(0, self.warmup_phase_epochs)
            self.memory.clear()
            return True
        return False
