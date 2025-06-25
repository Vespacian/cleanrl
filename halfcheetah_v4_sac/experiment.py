import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import gymnasium as gym
import torch.optim as optim

# network implementation
from actor_impl import Actor
from helper_functions import batch, run_eval

env_id = "HalfCheetah-v4"


# plots the reward over time
def plot(rewards, eval_freq, batch_size, lr, N, save_plot=False):
    x = np.arange(len(rewards)) * eval_freq
    plt.figure()
    plt.plot(x, rewards)
    plt.xlabel("training batches")
    plt.ylabel("rewards")
    plt.title(f'bs {batch_size}, lr {lr}, eval_freq {eval_freq}, N {N}')
    plt.grid(True)
    if save_plot:
        os.makedirs('plots', exist_ok=True)
        fname = f'halfcheetah_v4_sac/plots/bs{batch_size}_lr{lr}_ef{eval_freq}_N{N}.png'
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()
    print(f'plotted bs {batch_size}, lr {lr}, N {N}')


def train(data, weights, device, batch_size, lr, eval_freq, N):
    print("started training")
    # init
    env = gym.make(env_id)
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    actor = Actor(env).to(device)
    actor.load_state_dict(weights)
    optimizer = optim.Adam(actor.parameters(), lr=lr)
    
    # training loop
    actor.train()
    rewards = []
    batched_data = batch(data, batch_size=batch_size)
    print("entering training loop")
    for i, b in enumerate(batched_data):
        states = torch.stack([s for s, _ in b]).float().to(device=device)
        actions = torch.stack([a for _, a in b]).float().to(device=device)
        
        mean, log_std = actor(states)
        mean = mean * actor.action_scale + actor.action_bias
        std = log_std.exp() * actor.action_scale
        
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions)
        loss = -log_probs.sum(dim=1).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # eval
        if i % eval_freq == 0:
            # print(f'eval batch {i}')
            actor.eval()
            rewards.append(run_eval(actor, env, device, N))
            actor.train()
    
    return rewards




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load("halfcheetah_v4_sac/halfcheetah_v4_data.pt", map_location=device)
    weights = torch.load("halfcheetah_v4_sac/halfcheetah_v4_actor_weights.pt", map_location=device)
    
    # hparams
    batch_size = 2000
    lr = 1e-4
    eval_freq = 10
    N = 10
    
    rewards = train(data, weights, device, batch_size, lr, eval_freq, N)
    plot(rewards, eval_freq, batch_size, lr, N, False)
    
    # batch_size = [1000, 2000]
    # lr = [1e-3, 1e-4]
    # eval_freq = 10
    # N = [5, 10]
    
    # for bs in batch_size:
    #     for learn in lr:
    #         for n in N:
    #             rewards = train(data, weights, device, bs, learn, eval_freq, n)
    #             plot(rewards, eval_freq, bs, learn, n, True)
    