import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from const import const_var
from agents.controller import DeepLearningController, get_param_or_default
from modules import MLP, AdversarialModule, MLP_REPEAT
import time
from GAT.models import GAT


class PPO_con_Pi_Net(nn.Module):
    def __init__(self, input_size, use_gat=True):
        super(PPO_con_Pi_Net, self).__init__()
        self.use_gat = use_gat
        self.gat_net = GAT(nfeat=input_size,
                           nhid=8, nclass=1, dropout=0.6, nheads=3, alpha=0.2)
        self.net = nn.Sequential(
            nn.Linear(input_size * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
        )
        self.mu = nn.Linear(8, 8)
        self.sigma = nn.Linear(8, 8)
        self.value = nn.Linear(8, 1)

    def forward(self, x, adj):
        if self.use_gat:
            extract_x = self.gat_net(x, adj)
            # extract_x = extract_x.T[0]
            extract_x = torch.squeeze(extract_x, -1)
        else:
            # flatten_x = x.flatten()
            flatten_x = x.view(-1, 8*30)
            extract_x = self.net(flatten_x)
            extract_x = torch.squeeze(extract_x, 0)
        mu = torch.sigmoid(self.mu(extract_x))  # (0, 1)
        sigma = F.softplus(self.sigma(extract_x)) + 0.001
        return [mu, sigma], self.value(extract_x)


class PPO_con_Learner(DeepLearningController):

    def __init__(self, params):
        super(PPO_con_Learner, self).__init__(params)
        self.nr_epochs = get_param_or_default(params, "nr_epochs", 5)
        self.nr_episodes = get_param_or_default(params, "nr_episodes", 10)
        self.eps_clipping = get_param_or_default(params, "eps_clipping", 0.2)
        self.use_q_values = get_param_or_default(params, "use_q_values", False)
        self.warmup_phase_epochs = 50
        history_length = self.max_history_length
        input_shape = self.input_shape
        nr_actions = self.nr_actions
        # input_shape, outputs, max_history_length
        network_constructor = lambda in_shape, actions, length: PPO_con_Pi_Net(
            const_var.CELL_NUM_JUNCTION + const_var.CELL_NUM_LANE + 1, params['use_gat'])
        self.policy_net = AdversarialModule(input_shape, nr_actions, history_length, network_constructor).to(
            self.device)
        self.protagonist_optimizer = torch.optim.Adam(self.policy_net.protagonist_parameters(), lr=self.alpha)

    def joint_action_probs(self, histories, adj, training_mode=True, agent_ids=None):
        history = torch.tensor(histories, device=self.device, dtype=torch.float32)
        # in_time = time.time()
        action_probs, value = self.policy_net(history, adj)
        # len_probs = len(probs)
        # print("1Once time is %s" % (time.time() - in_time))
        # assert len(probs) == 1, "Expected length 1, but got shape {}".format(probs.shape)
        # action_probs = probs.detach().cpu().numpy()[0]

        # value = value.detach()
        if training_mode and numpy.random.rand() <= self.epsilon:
            [mu, sigma] = action_probs
            mu_probs = numpy.random.random(mu.size())
            sigma_probs = numpy.random.random(sigma.size())
            probs = []
            for mu_item, sigma_itme in zip(mu_probs, sigma_probs):
                probs.append([mu_item, sigma_itme])
            return probs
        else:
            [mu, sigma] = action_probs
            probs = []
            for mu_item, sigma_item in zip(mu.detach().cpu().numpy(), sigma.detach().cpu().numpy()):
                probs.append([mu_item, sigma_item])
            return probs

    def policy_update(self, minibatch_data, optimizer, random_agent_indices=None):
        last_states = minibatch_data["last_states"]
        last_actions = minibatch_data["last_actions"]
        last_actions_prob = minibatch_data["last_actions_prob"]
        rewards = minibatch_data["rewards"]
        states = minibatch_data["states"]
        action_probs, expected_values = self.policy_net(last_states, self.adj)
        action_probs = torch.stack(action_probs, -1)

        policy_losses = []
        value_losses = []
        for probs, action, value, R, old_prob in zip(action_probs, last_actions, expected_values, rewards,
                                                     last_actions_prob):
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
        # new_dis = torch.distributions.normal.Normal(probs[0], probs[1])
        # old_dis = torch.distributions.normal.Normal(old_prob[0], old_prob[1])
        # log_prob_new = new_dis.log_prob(action)
        # log_prob_old = old_dis.log_prob(action)
        action = action.unsqueeze(1)
        if self.params['add_loss']:
            loss1 = torch.tensor(0.0).to(self.device)
            loss2 = torch.tensor(0.0).to(self.device)
        else:
            loss1 = torch.tensor([]).to(self.device)
            loss2 = torch.tensor([]).to(self.device)

        for i in range(probs.size(-2)):
            new_dis = torch.distributions.normal.Normal(probs[i][0], probs[i][1])
            lose = new_dis.log_prob(action[i])
            if self.params['add_loss']:
                loss1 += lose[0]
            else:
                loss1 = torch.cat((loss1, lose), 0)

        for i in range(old_prob.size(-2)):
            old_dis = torch.distributions.normal.Normal(old_prob[i][0], old_prob[i][1])
            lose = old_dis.log_prob(action[i])
            if self.params['add_loss']:
                loss2 += lose[0]
            else:
                loss2 = torch.cat((loss2, lose), 0)

        # ratio = torch.exp(log_prob_new - log_prob_old)
        ratio = torch.exp(torch.clamp(loss1 - loss2, -5, 5))
        # m1 = Categorical(probs)
        # m2 = Categorical(old_prob)
        # ratio = torch.exp(m1.log_prob(action) - m2.log_prob(action))
        clipped_ratio = torch.clamp(ratio, 1 - self.eps_clipping, 1 + self.eps_clipping)
        surrogate_loss1 = ratio * advantage
        surrogate_loss2 = clipped_ratio * advantage
        return -torch.min(surrogate_loss1, surrogate_loss2)

    def value_update(self, minibatch_data):
        pass

    def update(self, last_state, last_action, reward, state):
        super(PPO_con_Learner, self).update(last_state, last_action, reward, state)
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
