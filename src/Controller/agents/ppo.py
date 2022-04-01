import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import const
from agents.controller import DeepLearningController, get_param_or_default
from modules import MLP, AdversarialModule, MLP_REPEAT
import time
from GAT.models import GAT

class PPONet(nn.Module):
    def __init__(self, input_shape, nr_actions, max_history_length, q_values=False, use_gat=True):
        super(PPONet, self).__init__()
        num_head = 3
        num_action = 2
        # self.fc_net = MLP(input_shape, max_history_length)
        # self.gat_net = GAT(nfeat=2 * (const.const_var.CELL_NUM_JUNCTION + const.const_var.CELL_NUM_LANE) + 1,
        #                    nhid=8, nclass=num_action, dropout=0.6, nheads=num_head, alpha=0.2)
        self.use_gat = use_gat
        if self.use_gat:
            self.gat_net = GAT(nfeat= const.const_var.CELL_NUM_JUNCTION + const.const_var.CELL_NUM_LANE + 1,
                               nhid=8, nclass=num_action, dropout=0.6, nheads=num_head, alpha=0.2)
        else:
            self.no_gat_net = MLP_REPEAT([8, const.const_var.CELL_NUM_JUNCTION + const.const_var.CELL_NUM_LANE + 1],
                                         max_history_length, nr_hidden_units=32)
        # self.gat_net = GAT(nfeat=2*(const.const_var.CELL_NUM_LANE), nhid=8, nclass=num_action,
        #                    dropout=0.6, nheads=num_head, alpha=0.2)

        self.forward_net = MLP(input_shape, max_history_length, nr_hidden_units=16)
        # self.action_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions)
        self.action_head = nn.Linear(16, nr_actions)
        if q_values:
            # self.value_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions)
            # self.value_head = nn.Linear(64, nr_actions)
            self.value_head = nn.Linear(16, nr_actions)
        else:
            # self.value_head = nn.Linear(self.fc_net.nr_hidden_units, 1)
            # self.value_head = nn.Linear(64, 1)
            self.value_head = nn.Linear(16, 1)
        self.action_head_sigmoid = nn.Sigmoid()

    def forward(self, x, adj, use_gumbel_softmax=False):
        if self.use_gat:
            x_gat = self.gat_net(x, adj)
        else:
            x_gat = self.no_gat_net(x)
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
        network_constructor = lambda in_shape, actions, length: PPONet(in_shape, actions, length, self.use_q_values, params['use_gat'])
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
        # m1 = Categorical(probs)
        # m2 = Categorical(old_prob)
        action[action>=0.5] = 1
        action[action<0.5] = 0
        probs = probs.unsqueeze(1)
        probs_add = 1 - probs
        probs = torch.cat((probs, probs_add), 1)
        old_prob = old_prob.unsqueeze(1)
        old_prob_add = 1 - old_prob
        old_prob = torch.cat((old_prob, old_prob_add), 1)

        action = action.unsqueeze(1)
        if self.params['add_loss']:
            loss1 = torch.tensor(0.0).to(self.device)
            loss2 = torch.tensor(0.0).to(self.device)
        else:
            loss1 = torch.tensor([]).to(self.device)
            loss2 = torch.tensor([]).to(self.device)
        for i in range(int(probs.size(-2))):
            m = Categorical(probs[i])
            lose = m.log_prob(action[i])
            if self.params['add_loss']:
                loss1 += lose[0]
            else:
                loss1 = torch.cat((loss1, lose), 0)
        for i in range(int(old_prob.size(-2))):
            m = Categorical(old_prob[i])
            lose = m.log_prob(action[i])
            if self.params['add_loss']:
                loss2 += lose[0]
            else:
                loss2 = torch.cat((loss2, lose), 0)
        # ratio = torch.exp(m1.log_prob(action) - m2.log_prob(action))
        ratio = torch.exp(loss1 - loss2)
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
