import numpy as np
import matplotlib.pyplot as plt
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
                action = action.squeeze(0).cpu().numpy()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
            total += reward
    
    return total / N

# plots the reward over time
def plot(rewards, eval_freq):
    x = np.arange(len(rewards)) * eval_freq
    plt.figure()
    plt.plot(x, rewards)
    plt.xlabel("training batches")
    plt.ylabel("rewards")
    plt.grid(True)
    plt.show()