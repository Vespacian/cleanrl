
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class DiffusionActor(nn.Module):
    def __init__(self, env, scheduler, T=25):
        super().__init__()
        self.T = T
        self.scheduler = scheduler
        self.state_dim = np.prod(env.single_observation_space.shape)
        self.action_dim = np.prod(env.single_action_space.shape)
        
        hidden_dim = 512
        self.time_embed = nn.Embedding(T, hidden_dim)
        
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim)
        )     
        
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

    def forward(self, state, action_noise, t):
        time_embed = self.time_embed(t)
        x = torch.cat([state, action_noise, time_embed], 1)
        return self.net(x)

    def get_action(self, state, device):
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(state).float().to(device).unsqueeze(0)
            a = torch.randn(1, self.action_dim, device=device)
            
            for t in self.scheduler.timesteps:
                t_batch = torch.full((1,), int(t), dtype=torch.long, device=device)
                pred = self.forward(x, a, t_batch)
                step = self.scheduler.step(pred, int(t), a)
                a = step.prev_sample
        
        self.train()
        a = torch.tanh(a) * self.action_scale + self.action_bias
        return a.squeeze(0).cpu().numpy()
    
    # get action per batch for vectorization
    def get_actions_batch(self, obs, device):
        with torch.no_grad():
            x = torch.from_numpy(obs).float().to(device)
            B = x.shape[0]
            a = torch.randn(B, self.action_dim, device=device)
            
            for t in self.scheduler.timesteps:
                t_batch = torch.full((B,), int(t), dtype=torch.long, device=device)
                pred = self.forward(x, a, t_batch)
                step = self.scheduler.step(pred, int(t), a)
                a = step.prev_sample
        
        a = torch.tanh(a) * self.action_scale + self.action_bias
        return a.cpu().numpy()


# the unchanged version in the given code
class OGActor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
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

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean



# MSE
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        
        self.K = 5
        self.action_dim = np.prod(env.single_action_space.shape)
        self.fc_mean = nn.Linear(256, self.K * self.action_dim)
        self.fc_logstd = nn.Linear(256, self.K * self.action_dim)
        self.fc_logits = nn.Linear(256, self.K)
        
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

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        feat = F.relu(self.fc3(x))
        
        logits = self.fc_logits(feat)
        
        mean = self.fc_mean(feat)
        mean = mean.view(-1, self.K, self.action_dim)
        
        log_std = self.fc_logstd(feat)
        log_std = log_std.view(-1, self.K, self.action_dim)
        log_std = torch.tanh(log_std)
        
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return logits, mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    



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