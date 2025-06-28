import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import gymnasium as gym
import torch.optim as optim

# network implementation
from actor_impl import Actor
from helper_functions import batch, run_eval, run_eval_gauss
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

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
        os.makedirs('halfcheetah_v4_sac/plots', exist_ok=True)
        fname = f'halfcheetah_v4_sac/plots/bs{batch_size}_lr{lr}_ef{eval_freq}_N{N}.png'
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()
    print(f'plotted bs {batch_size}, lr {lr}, N {N}')


def train(data, weights, device, batch_size, lr, eval_freq, N, weight_decay, epochs=3):
    print("started training")
    # init
    env = gym.make(env_id)
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    actor = Actor(env).to(device)
    
    new_sd = actor.state_dict()

    D = actor.action_dim  # e.g. 6
    K = actor.K           # e.g. 5
    for k, v in weights.items():
        if k in new_sd and v.shape == new_sd[k].shape:
            new_sd[k] = v
        
        if k == "fc_mean.weight":
            new_sd[k][:D, :] = v
        elif k == "fc_mean.bias":
            new_sd[k][:D] = v
        elif k == "fc_logstd.weight":
            new_sd[k][:D, :] = v
        elif k == "fc_logstd.bias":
            new_sd[k][:D] = v
    
    actor.load_state_dict(new_sd)
    optimizer = optim.Adam(actor.parameters(), lr=lr, weight_decay=weight_decay)
    
    # training loop
    actor.train()
    rewards = []
    batched_data = batch(data, batch_size=batch_size)
    print("entering training loop")
    for _ in range(epochs):
        for i, b in enumerate(batched_data):
            states = torch.stack([s for s, _ in b]).float().to(device=device)
            actions = torch.stack([a for _, a in b]).float().to(device=device)
            
            logits, mean, log_std = actor(states)
            mean = mean * actor.action_scale + actor.action_bias
            std = log_std.exp() * actor.action_scale
            
            # mix = Categorical(logits=logits)
            # comp = Independent(Normal(mean, std), 1)
                    
            # dist = MixtureSameFamily(mix, comp)
            # log_probs = dist.log_prob(actions)
            
            probs = torch.softmax(logits, dim=1)
            mix = (probs.unsqueeze(-1) * mean).sum(dim=1)
            pred = torch.tanh(mix) * actor.action_scale + actor.action_bias
            
            # confirm that data loaded correctly
            if i == 0:
                print("MoG shapes:", logits.shape, mean.shape, std.shape)
                print("Mixture weights (first):", torch.softmax(logits,1)[0].detach().cpu().numpy())
                print("Means[0,k,:] for k=0..4:", mean[0].detach().cpu().numpy())  
                print("Std[0,k,:] :", std[0].detach().cpu().numpy())
            
            # loss = -log_probs.mean()
            loss = F.mse_loss(pred, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # eval
            if i % eval_freq == 0:
                # print(f'eval batch {i}')
                actor.eval()
                rewards.append(run_eval_gauss(actor, env, device, N))
                actor.train()
    
    print(f'final reward: {rewards[-1]}')
    
    return rewards




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load("halfcheetah_v4_sac/halfcheetah_v4_data.pt", map_location=device)
    weights = torch.load("halfcheetah_v4_sac/halfcheetah_v4_actor_weights.pt", map_location=device)
    
    # hparams
    batch_size = 500
    lr = 1e-3
    eval_freq = 10
    N = 30
    epochs = 3
    weight_decay = 1e-6
    
    rewards = train(
        data=data, 
        weights=weights, 
        device=device, 
        batch_size=batch_size, 
        lr=lr, 
        eval_freq=eval_freq, 
        N=N, 
        weight_decay=weight_decay,
        epochs=epochs
    )
    plot(rewards, eval_freq, batch_size, lr, N, True)
    
    # batch_size = [500, 1000, 2000, 4000]
    # lr = [1e-3, 1e-4, 5e-4]
    # eval_freq = 10
    # N = 10
    # epochs = 1
    
    # for bs in batch_size:
    #     for l in lr:
    #         rewards = train(data, weights, device, bs, l, eval_freq, N, epochs=epochs)
    #         plot(rewards, eval_freq, bs, l, N, True)
    