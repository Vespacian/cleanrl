import numpy as np
import torch

# returns num_split amount of batches
# in each batch, this will return a list of (s, a) elements
def batch(data, batch_size=500):
    num_splits = len(data['observations']) // batch_size
    
    obs = np.array_split(data['observations'], num_splits)
    actions = np.array_split(data['actions'], num_splits)
    
    return [list(zip(o, a)) for o, a in zip(obs, actions)]

# diffusion time helper
def make_onehot(t, T, device):
    onehot = torch.zeros(t.shape[0], T, device=device)
    return onehot.scatter_(1, t.unsqueeze(1), 1.0)

# scheduler for beta
def make_schedule(T, start, end, device="cpu"):
    return torch.linspace(start, end, T, device=device)

def run_eval_diff(model, stats, scheduler, env, device, N=30):
    mean, std = stats
    total = 0
    for _ in range(N):
        obs, info = env.reset()
        episode_over = False
        while not episode_over:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            obs_tensor = (obs_tensor - mean) / std
            x = torch.randn(1, env.single_action_space.shape[0], device=device)
            
            for t in reversed(range(scheduler.T)):
                val = torch.full((1,), t, dtype=torch.long, device=device)
                x = scheduler.p_sample(model, x, val, obs_tensor, device)
            
            action = (x * model.action_scale + model.action_bias)
            action = action.clamp(env.single_action_space.low[0], env.single_action_space.high[0]).cpu().squeeze(0).detach().numpy()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
            total += reward
    
    return total / N


def run_eval(actor, env, device, N=5):
    total = 0
    for _ in range(N): 
        obs, info = env.reset()
        episode_over = False
        
        while not episode_over:
            with torch.no_grad():
                obs_tensor = (torch.from_numpy(obs).float().unsqueeze(0).to(device))
                action, log_prob, mean = actor.get_action(obs_tensor)
                mean = mean.squeeze(0).cpu().numpy()
            
            obs, reward, terminated, truncated, info = env.step(mean)
            episode_over = terminated or truncated
            total += reward
    
    return total / N

def run_eval_gauss(actor, env, device, N=5):
    total = 0
    for _ in range(N): 
        obs, info = env.reset()
        episode_over = False
        
        while not episode_over:
            with torch.no_grad():
                obs_tensor = (torch.from_numpy(obs).float().unsqueeze(0).to(device))
                logits, mean, logstd = actor(obs_tensor)
            
            probs = torch.softmax(logits, dim=1)
            mix = (probs.unsqueeze(-1) * mean).sum(dim=1)
            mean = torch.tanh(mix) * actor.action_scale + actor.action_bias
            mean = mean.squeeze(0).cpu().numpy()
            
            obs, reward, terminated, truncated, info = env.step(mean)
            episode_over = terminated or truncated
            total += reward
    
    return total / N

def collect_states(actor, env, device, N=5):
    states = []
    for _ in range(N): 
        obs, info = env.reset()
        episode_over = False
        
        while not episode_over:
            with torch.no_grad():
                obs_tensor = (torch.from_numpy(obs).float().unsqueeze(0).to(device))
                action, log_prob, mean = actor.get_action(obs_tensor)
                mean = mean.squeeze(0).cpu().numpy()
            
            obs, reward, terminated, truncated, info = env.step(mean)
            episode_over = terminated or truncated
            states.append(obs.copy())
    
    return np.array(states)

def collect_actions(actor, states, device):
    with torch.no_grad():
        obs_tensor = torch.from_numpy(states).float().to(device)
        action, log_prob, mean = actor.get_action(obs_tensor)
    return mean.cpu().numpy()