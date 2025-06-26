import numpy as np
import torch

# returns num_split amount of batches
# in each batch, this will return a list of (s, a) elements
def batch(data, batch_size=500):
    num_splits = len(data['observations']) // batch_size
    
    obs = np.array_split(data['observations'], num_splits)
    actions = np.array_split(data['actions'], num_splits)
    
    return [list(zip(o, a)) for o, a in zip(obs, actions)]


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