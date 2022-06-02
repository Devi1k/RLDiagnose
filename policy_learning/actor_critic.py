import os
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch import optim as optim
from torch.distributions import Categorical


class ActorCritic(object):
    def __init__(self, n_features, n_actions, param):
        self.device = torch.device('cuda',
                                   index=param.get("gpu_index", 1)) if torch.cuda.is_available() else torch.device(
            'cpu')
        self.actor = Actor(n_features=n_features, n_actions=n_actions, lr=param.get("LR_A", 0.001), param=param).to(
            self.device)
        self.critic = Critic(n_features=n_features, lr=param.get("LR_C", 0.01), param=param).to(self.device)
        self.gamma = param.get("GAMMA")
        self.Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                                    'reward'))

        if param["train_mode"] != 1:
            self.restore_model(param["saved_model"])
            self.actor.eval()
            # self.target_net.eval()

    def train(self, trajectories):
        b_obs, b_acts, b_mask, b_next_obs, b_r = [], [], [], [], []

        for trajectory in trajectories:
            obs, actions, mask, next_obs, r = self.__process_trajectory__(trajectory)
            b_obs.extend(obs)
            b_acts.extend(actions)
            b_mask.extend(mask)
            b_next_obs.extend(next_obs)
            b_r.extend(r)

        td_error = self.critic.learn(b_obs, b_r, b_next_obs, b_mask)  # gradient = grad[r + gamma * V(s_) - V(s)]
        self.actor.learn(b_obs, b_acts, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

    def __process_trajectory__(self, trajectory):
        # state, agent_action, reward, next_state, episode_over

        obs, acts, r, next_obs, mask = [], [], [], [], []
        for turn in trajectory:
            obs.append(turn[0])
            acts.append(turn[1])
            r.append(turn[2])

            next_obs.append(turn[3])
            mask.append(turn[4])


        return list(obs), list(acts), list(mask), list(next_obs), list(r)

    def restore_model(self, saved_model):
        """
        Restoring the trained parameters for the model. Both current and target net are restored from the same parameter.

        Args:
            saved_model (str): the file name which is the trained model.
        """
        print("loading trained model", saved_model)
        if torch.cuda.is_available() is False:
            map_location = 'cpu'
        else:
            map_location = None
        self.actor.load_state_dict(torch.load(saved_model, map_location=map_location))
        # self.target_net.load_state_dict(self.current_net.state_dict())

    def save_model(self, model_performance, episodes_index, checkpoint_path):
        if os.path.isdir(checkpoint_path) is False:
            # os.mkdir(checkpoint_path)
            # print(os.getcwd())
            os.makedirs(checkpoint_path)
        # agent_id = self.params.get("agent_id").lower()
        # disease_number = self.params.get("disease_number")
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        average_wrong_disease = model_performance["average_wrong_disease"]
        model_file_name = os.path.join(checkpoint_path,
                                       "model_d" + "_agent" + "_a2c" + "_s" + str(success_rate) + "_r" + str(
                                           average_reward) + "_t" + str(average_turn) + "_wd" + str(
                                           average_wrong_disease) + "_e" + str(episodes_index) + ".pkl")

        torch.save(self.actor.state_dict(), model_file_name)


class Actor(nn.Module):
    def __init__(self, n_features, n_actions, param, hidden_size=(128, 128), lr=0.0001):
        super().__init__()
        self.activation = torch.relu
        self.affine_layers = nn.ModuleList()
        last_dim = n_features
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh
        self.action_head = nn.Linear(last_dim, n_actions)
        self.optim = optim.Adam(self.parameters(), lr)
        self.device = torch.device('cuda',
                                   index=param.get("gpu_index", 1)) if torch.cuda.is_available() else torch.device(
            'cpu')

    def forward(self, states):
        # s = torch.tensor(s).to(torch.float32).to(self.device)
        s = torch.from_numpy(np.stack(states)).to(torch.float32).to(self.device).unsqueeze(0)
        for affine in self.affine_layers:
            s = self.activation(affine(s))
        action_prob = torch.softmax(self.action_head(s), dim=1)
        return action_prob

    def learn(self, states, actions, td_error):
        # actions = torch.from_numpy(np.stack(actions)).to(torch.float64).to(self.device)

        log_probs = self.get_log_prob(states)
        policy_loss = -(log_probs * td_error).mean()
        self.optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 40)
        self.optim.step()

    def get_kl(self, x):
        action_prob1 = self.forward(x)
        action_prob0 = action_prob1.detach()
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x):
        action_prob = self.forward(x)
        action_distribution = Categorical(action_prob)
        action = action_distribution.sample()
        return action_distribution.log_prob(action)

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}

    def select_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.multinomial(1)
        return action


class Critic(nn.Module):
    def __init__(self, n_features, param, lr=0.01, gamma=0.9, l2_reg=1e-3, hidden_size=(128, 128), **kwargs):
        super().__init__(**kwargs)
        # self.l1 = nn.Linear(n_features, 30)  # relu
        # self.v = nn.Linear(30, 1)  #
        self.affine_layers = nn.ModuleList()
        last_dim = n_features
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        self.activation = torch.relu
        self.optim = optim.Adam(self.parameters(), lr)
        self.gamma = gamma
        self.l2_reg = l2_reg
        self.loss = torch.nn.MSELoss()
        self.device = torch.device('cuda',
                                   index=param.get("gpu_index", 1)) if torch.cuda.is_available() else torch.device(
            'cpu')

    def forward(self, s):
        for affine in self.affine_layers:
            s = self.activation(affine(s))

        value = self.value_head(s)
        return value

    def learn(self, s, r, s_, masks):
        s = torch.from_numpy(np.stack(s)).to(torch.float32).to(self.device).squeeze()
        r = torch.from_numpy(np.stack(r)).to(torch.float32).to(self.device).unsqueeze(1)
        s_ = torch.from_numpy(np.stack(s_)).to(torch.float32).to(self.device).squeeze()
        masks = torch.from_numpy(np.stack(masks)).to(torch.float32).to(self.device).unsqueeze(1)

        with torch.no_grad():
            v_ = self.forward(s_).squeeze(0)
            v = self.forward(s).squeeze(0)
        td_target = r + self.gamma * torch.mul(v_, masks)
        advantages = r + self.gamma * torch.mul(v_, masks) - v
        # advantages, td_target = [], []
        # for i in range(r.size(0)):
        #     advantages.append(r[i] + self.gamma * v_[i] * masks[i] - v[i])
        #     td_target.append(r[i] + self.gamma * v_[i] * masks[i])
        # td_target = torch.from_numpy(np.stack(td_target)).to(torch.float64).to(self.device)
        loss = self.loss(v, td_target)
        # for param in self.parameters():
        #     loss += param.pow(2).sum() * self.l2_reg
        loss.requires_grad_(True)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        return advantages
