import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import gymnasium as gym

# network implementation
from actor_impl import Actor

env_id = "HalfCheetah-v4"

def load_pt(file_path: str, device):
    data = torch.load(file_path, map_location=device)
    return data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = load_pt("halfcheetah_v4_sac/halfcheetah_v4_data.pt", device=device)
    weights = load_pt("halfcheetah_v4_sac/halfcheetah_v4_actor_weights.pt", device=device)
    print(data)
    print()
    print(weights)
    print()
    print(len(weights['fc_mean.weight'][0]))
    