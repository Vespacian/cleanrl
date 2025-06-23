import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import gymnasium as gym

# network implementation
from actor_impl import Actor
from helper_functions import batch

env_id = "HalfCheetah-v4"

def train(data, weights, device):
    # init
    env = gym.make(env_id)
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    actor = Actor(env).to(device)
    actor.load_state_dict(weights)
    
    # training loop
    batched_data = batch(data, batch_size=500)
    
    for b in batched_data:
        for s, a in b:
            # for each s, take predicted mean action
            pass



def plot_reward_over_time():
    pass

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load("halfcheetah_v4_sac/halfcheetah_v4_data.pt", map_location=device)
    weights = torch.load("halfcheetah_v4_sac/halfcheetah_v4_actor_weights.pt", map_location=device)
    
    train(data, weights, device)
    
    
    