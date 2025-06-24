import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import gymnasium as gym
import torch.optim as optim

# network implementation
from actor_impl import Actor
from helper_functions import batch, run_eval, plot

# variables and hparams
env_id = "HalfCheetah-v4"
batch_size = 1000
lr = 1e-3
eval_freq = 10
N = 5
# epochs = 10


def train(data, weights, device):
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
    
    rewards = train(data, weights, device)
    print("plotting")
    plot(rewards, eval_freq)
    
    
    