from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch import optim as optim
from torch.distributions import Categorical


class ActorCritic(object):
    def __init__(self, n_features, n_actions, param, ):
        self.actor = Actor(n_features=n_features, n_actions=n_actions, lr=param.get("LR_A", 0.001), param=param)
        self.critic = Critic(n_features=n_features, lr=param.get("LR_C", 0.01), param=param)
        self.gamma = param.get("GAMMA")
        self.Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                                    'reward'))

    def train(self, trajectories):
        b_obs, b_acts, b_mask, b_next_obs, b_r = [], [], [], [], []

        for trajectory in trajectories:
            obs, actions, mask, next_obs, r = self.__process_trajectory__(trajectory)
            b_obs.extend(obs)
            b_acts.extend(actions)
            b_mask.extend(mask)
            b_next_obs.extend(next_obs)
            b_r.extend(r)

        td_error = self.critic.learn(b_obs, b_r, b_next_obs, b_mask).item()  # gradient = grad[r + gamma * V(s_) - V(s)]
        self.actor.learn(b_obs, b_acts, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

    def __process_trajectory__(self, trajectory):
        # state, agent_action, reward, next_state, episode_over

        obs, acts, r, next_obs, mask = [], [], [], [], []
        for turn in trajectory:
            obs.append(turn[0])
            acts.append(turn[1])
            mask.append(turn[2])
            next_obs.append(turn[3])
            r.append(turn[4])

        return list(obs), list(acts), list(mask), list(next_obs), list(r)


class Actor(nn.Module):
    def __init__(self, n_features, n_actions, param, hidden_size=(128, 128), lr=0.0001, **kwargs):
        super().__init__(**kwargs)
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

    def forward(self, s):
        s = torch.tensor(s, dtype=torch.float32, device=next(self.parameters()).device)
        for affine in self.affine_layers:
            s = self.activation(affine(s))
        action_prob = torch.softmax(self.action_head(s), dim=0)
        return action_prob

    def learn(self, states, actions, td_error):
        states = torch.from_numpy(np.stack(states)).to(torch.float64).to(self.device)
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
        action_distribution = Categorical(action_prob)
        action = action_distribution.sample()
        return action


class Critic(nn.Module):
    def __init__(self, n_features, param, lr=0.01, gamma=0.9, l2_reg=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.l1 = nn.Linear(n_features, 30)  # relu
        self.v = nn.Linear(30, 1)  #
        self.optim = optim.Adam(self.parameters(), lr)
        self.gamma = gamma
        self.l2_reg = l2_reg
        self.device = torch.device('cuda',
                                   index=param.get("gpu_index", 1)) if torch.cuda.is_available() else torch.device(
            'cpu')

    def forward(self, s):
        s = torch.tensor(s, dtype=torch.float32, device=next(self.parameters()).device)[None]
        return self.v(self.l1(s).relu())

    def learn(self, s, r, s_, masks):
        s = torch.from_numpy(np.stack(s)).to(torch.float64).to(self.device)
        r = torch.from_numpy(np.stack(r)).to(torch.float64).to(self.device)
        s_ = torch.from_numpy(np.stack(s_)).to(torch.float64).to(self.device)
        masks = torch.from_numpy(np.stack(masks)).to(torch.float64).to(self.device)

        with torch.no_grad():
            v_ = self.forward(s_).squeeze()
            v = self.forward(s).squeeze()
        # tensor_type = type(r)
        # advantages = tensor_type(r.size(0), 1)
        advantages = []
        for i in range(r.size(0)):
            advantages.append(r[i] + self.gamma * v_[i] * masks[i] - v[i])
        advantages = torch.from_numpy(np.stack(advantages)).to(torch.float64).to(self.device)
        advantages = (advantages - advantages.mean()) / advantages.std()
        td_error = torch.mean(advantages)
        loss = td_error.square()
        for param in self.parameters():
            loss += param.pow(2).sum() * self.l2_reg

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return td_error
