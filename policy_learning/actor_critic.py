import torch
from torch import nn


# class Actor(nn.Module):
#     def __init__(self, n_features, action_bound, lr=0.0001, **kwargs):
#         super().__init__(**kwargs)
#         self.action_bound = action_bound
#
#         self.l1 = nn.Linear(n_features, 30)  # relu
#         self.mu = nn.Linear(30, 1)  # tanh
#         self.sigma = nn.Linear(30, 1)  # log(exp(features) + 1)
#
#         self.normal_dist = distributions.Normal(0, 1)
#         self.optim = optim.Adam(self.parameters(), lr)
#
#     def forward(self, s):
#         s = torch.tensor(s, dtype=torch.float32, device=next(self.parameters()).device)[None]
#         h = self.l1(s).relu()
#         mu = self.mu(h).tanh()
#         sigma = self.sigma(h)
#         sigma = torch.log(sigma.exp() + 1)
#
#         self.normal_dist = distributions.Normal(mu[0] * 2, sigma[0] + .1)
#         action = self.normal_dist.sample()
#         action = torch.clip(action, self.action_bound[0], self.action_bound[1])
#         return action
#
#     def learn(self, action, td_error):
#         action_prob = self.normal_dist.log_prob(action)
#         exp_v = action_prob * td_error.detach() + 0.01 * self.normal_dist.entropy()
#         loss = -exp_v.sum()
#
#         loss.backward()
#         self.optim.step()
#         self.optim.zero_grad()
#         return exp_v
#
#
# class Critic(nn.Module):
#     def __init__(self, n_features, lr=0.01, **kwargs):
#         super().__init__(**kwargs)
#         self.l1 = nn.Linear(n_features, 30)  # relu
#         self.v = nn.Linear(30, 1)  #
#         self.optim = optim.Adam(self.parameters(), lr)
#
#     def forward(self, s):
#         s = torch.tensor(s, dtype=torch.float32, device=next(self.parameters()).device)[None]
#         return self.v(self.l1(s).relu())
#
#     def learn(self, s, r, s_):
#         with torch.no_grad():
#             v_ = self(s_)
#         td_error = torch.mean((r + GAMMA * v_) - self(s))
#         loss = td_error.square()
#
#         loss.backward()
#         self.optim.step()
#         self.optim.zero_grad()
#         return td_error


class Actor(nn.Module):
    def __init__(self, state_dim, action_num, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        self.is_disc_action = True
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_head = nn.Linear(last_dim, action_num)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_prob = torch.softmax(self.action_head(x), dim=1)
        return action_prob

    def select_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.multinomial(1)
        return action

    def get_kl(self, x):
        action_prob1 = self.forward(x)
        action_prob0 = action_prob1.detach()
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        return torch.log(action_prob.gather(1, actions.long().unsqueeze(1)))

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value
