
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from helper_functions import make_schedule, make_onehot

LOG_STD_MAX = 2
LOG_STD_MIN = -5

# Diffusion
class Actor(nn.Module):
    def __init__(self, env, hidden = 256, T=50):
        super().__init__()
        obs_dim = int(torch.tensor(env.single_observation_space.shape).prod())
        act_dim = int(torch.tensor(env.single_action_space.shape).prod())
        self.T = T
        
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden * 2)
        )
        
        self.time_mlp = nn.Sequential(
            nn.Linear(T, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        
        self.conv1 = nn.Conv1d(act_dim, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden, act_dim, kernel_size=3, padding=1)
        
        
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, actions, obs, onehot):
        gamma, beta = self.obs_mlp(obs).chunk(2, dim=-1)
        t = self.time_mlp(onehot)
        
        x = actions.unsqueeze(-1)
        
        x = self.conv1(x)
        x = F.relu(x)
        x = x * gamma.unsqueeze(-1) + beta.unsqueeze(-1)
        
        x = self.conv2(x)
        x = F.relu(x + t.unsqueeze(-1))
        
        x = self.conv3(x)
        
        return x.squeeze(-1)
        

    # def get_action(self, x):
    #     mean, log_std = self(x)
    #     std = log_std.exp()
    #     normal = torch.distributions.Normal(mean, std)
    #     x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    #     y_t = torch.tanh(x_t)
    #     action = y_t * self.action_scale + self.action_bias
    #     log_prob = normal.log_prob(x_t)
    #     # Enforcing Action Bound
    #     log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
    #     log_prob = log_prob.sum(1, keepdim=True)
    #     mean = torch.tanh(mean) * self.action_scale + self.action_bias
    #     return action, log_prob, mean

class Scheduler:
    def __init__(self, T=50, device="cpu"):
        self.T = T
        self.device = device
        self.betas = make_schedule(T, 1e-4, 2e-2, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, dim=0)
    
    def q_sample(self, x0, t, noise):
        prod = self.alphas_prod[t]
        return (prod.sqrt().unsqueeze(-1) * x0) + ((1-prod).sqrt().unsqueeze(-1) * noise)
    
    def p_sample(self, model, xt, t, obs, device):
        beta = self.betas[t].unsqueeze(-1)
        alpha = self.alphas[t].unsqueeze(-1)
        prod = self.alphas_prod[t].unsqueeze(-1)
        
        onehot = make_onehot(t, model.T, device)
        epsilon = model(xt, obs, onehot)
        
        pred = (xt - (beta / (1 - prod).sqrt()) * epsilon) / alpha.sqrt()
        mu = alpha.sqrt() * pred + (1-alpha).sqrt() * epsilon
        if t[0] > 0:
            noise = torch.randn_like(xt)
        else:
            noise = torch.zeros_like(xt)
        
        return mu + beta.sqrt() * noise
    
    
    
    
    
    
    # baseline
# class Actor(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
#         self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
#         # action rescaling
#         self.register_buffer(
#             "action_scale",
#             torch.tensor(
#                 (env.single_action_space.high - env.single_action_space.low) / 2.0,
#                 dtype=torch.float32,
#             ),
#         )
#         self.register_buffer(
#             "action_bias",
#             torch.tensor(
#                 (env.single_action_space.high + env.single_action_space.low) / 2.0,
#                 dtype=torch.float32,
#             ),
#         )

#     # baseline
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         mean = self.fc_mean(x)
#         log_std = self.fc_logstd(x)
#         log_std = torch.tanh(log_std)
#         log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
#             log_std + 1
#         )  # From SpinUp / Denis Yarats

#         return mean, log_std

#     def get_action(self, x):
#         mean, log_std = self(x)
#         std = log_std.exp()
#         normal = torch.distributions.Normal(mean, std)
#         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#         log_prob = normal.log_prob(x_t)
#         # Enforcing Action Bound
#         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
#         log_prob = log_prob.sum(1, keepdim=True)
#         mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         return action, log_prob, mean

# MSE
# class Actor(nn.Module):
    # def __init__(self, env):
    #     super().__init__()
    #     self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 512)
    #     self.fc2 = nn.Linear(512, 512)
    #     self.fc3 = nn.Linear(512, 256)
        
    #     self.K = 5
    #     self.action_dim = np.prod(env.single_action_space.shape)
    #     self.fc_mean = nn.Linear(256, self.K * self.action_dim)
    #     self.fc_logstd = nn.Linear(256, self.K * self.action_dim)
    #     self.fc_logits = nn.Linear(256, self.K)
        
    #     # action rescaling
    #     self.register_buffer(
    #         "action_scale",
    #         torch.tensor(
    #             (env.single_action_space.high - env.single_action_space.low) / 2.0,
    #             dtype=torch.float32,
    #         ),
    #     )
    #     self.register_buffer(
    #         "action_bias",
    #         torch.tensor(
    #             (env.single_action_space.high + env.single_action_space.low) / 2.0,
    #             dtype=torch.float32,
    #         ),
    #     )

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         feat = F.relu(self.fc3(x))
        
#         logits = self.fc_logits(feat)
        
#         mean = self.fc_mean(feat)
#         mean = mean.view(-1, self.K, self.action_dim)
        
#         log_std = self.fc_logstd(feat)
#         log_std = log_std.view(-1, self.K, self.action_dim)
#         log_std = torch.tanh(log_std)
        
#         log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
#             log_std + 1
#         )  # From SpinUp / Denis Yarats

#         return logits, mean, log_std

#     def get_action(self, x):
#         mean, log_std = self(x)
#         std = log_std.exp()
#         normal = torch.distributions.Normal(mean, std)
#         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#         log_prob = normal.log_prob(x_t)
#         # Enforcing Action Bound
#         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
#         log_prob = log_prob.sum(1, keepdim=True)
#         mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         return action, log_prob, mean