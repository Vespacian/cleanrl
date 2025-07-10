import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import gymnasium as gym
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# network implementation
from actor_impl import AutoregActor
from helper_functions import run_eval_autoreg_vec
import torch.nn.functional as F
from diffusers import DDPMScheduler
import tyro
from dataclasses import dataclass

env_id = "HalfCheetah-v4"

@dataclass
class Config:  
    batch_size: int = 512
    lr: float = 1e-3
    eval_freq: int = 4688
    eval_episodes: int = 16
    epochs: int = 40
    weight_decay: float = 1e-6
    num_env: int = 16
    
    # timesteps: int = 50
    logdir: str = None
    bins: int = 31



def train(data, weights, device, config: Config):
    start_time = time.time()
    print("started training")
    if config.logdir is None:
        logdir = f"halfcheetah_v4_sac/runs/autoreg/bs{config.batch_size}_lr{config.lr}_b{config.bins}"
        writer = SummaryWriter(log_dir=logdir)
    else:
        writer = SummaryWriter(log_dir=config.logdir)
    # init
    env = gym.make(env_id)
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    
    # actor = NewDiffusionActor(env, scheduler, timesteps=config.timesteps).to(device)
    actor = AutoregActor(env, config.bins).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # obs = torch.tensor(data['observations'], dtype=torch.float32).to(device)
    # act = torch.tensor(data['actions'], dtype=torch.float32).to(device)
    obs = data['observations'].detach().clone().float().to(device)
    act = data['actions'].detach().clone().float().to(device)
    dataset = TensorDataset(obs, act)
    batched_data = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    # batched_data = batch(data, batch_size=batch_size) 
    
    # training loop
    actor.train()
    # rewards = []
    step = 0
    print("entering training loop")
    for i in range(config.epochs):
        epoch_time = time.time()
        epoch_loss = 0
        for j, b in enumerate(batched_data):
            # states = torch.stack([s for s, _ in b]).float().to(device=device)
            # actions = torch.stack([a for _, a in b]).float().to(device=device)
            states = b[0].float().to(device)
            actions = b[1].float().to(device)
            
            # t = torch.randint(0, actor.timesteps, (states.size(0),), device=device)
            # noise = torch.randn_like(actions)
            # action_noise = scheduler.add_noise(actions, noise, t)
            
            # pred = actor(states, action_noise, t)
            # loss = F.mse_loss(pred, noise)
            
            logprob = actor.log_prob(states, actions)
            loss = -logprob.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            writer.add_scalar("Train/Loss", loss.item(), step)
            step += 1
            
            # eval
            if j % config.eval_freq == 0:
                eval_time = time.time()
                # actor.eval()
                # rewards.append(run_eval_diff_vec(actor, env_id, device, config.eval_episodes, num_env=config.num_env))
                # actor.train()
                reward = run_eval_autoreg_vec(actor, env_id, device, config.eval_episodes, num_env=config.num_env)
                print(f"Eval {j} time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - eval_time))}")
                writer.add_scalar("Eval/Reward", reward, step)
        
        avg_loss = epoch_loss / (len(data['observations']) / config.batch_size)
        print(f'Epoch {i + 1} MSE loss: {avg_loss}')
        print(f"Epoch {i + 1}/{config.epochs} time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_time))}")
    
    writer.close()
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    # print(f'final reward: {rewards[-1]}')
    
    # return rewards 

# python halfcheetah_v4_sac/experiment.py
# tensorboard --logdir halfcheetah_v4_sac/runs
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load("halfcheetah_v4_sac/halfcheetah_v4_data.pt", map_location=device, weights_only=False)
    weights = torch.load("halfcheetah_v4_sac/halfcheetah_v4_actor_weights.pt", map_location=device)
        
    config = tyro.cli(Config)
    print(f'config: {config}')
    
    train(
        data=data, 
        weights=weights, 
        device=device, 
        config=config, 
    )
    
    
   
