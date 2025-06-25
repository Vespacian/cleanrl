import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import gymnasium as gym
import torch.optim as optim

# network implementation
from actor_impl import Actor
from helper_functions import batch, run_eval, collect_states, collect_actions

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


def train(data, optimizer, device, env, actor, batch_size, eval_freq, N, epochs=3):
    print("started training")
    
    # training loop
    actor.train()
    rewards = []
    batched_data = batch(data, batch_size=batch_size)
    print("entering training loop")
    for _ in range(epochs):
        for i, b in enumerate(batched_data):
            states = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s, _ in b]).to(device=device)
            actions = torch.stack([torch.as_tensor(a, dtype=torch.float32) for _, a in b]).to(device=device)
            
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
    epochs = 3
    
    # init
    env = gym.make(env_id)
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    
    expert = Actor(env).to(device)
    expert.load_state_dict(weights)
    policy = Actor(env).to(device)
    policy.load_state_dict(weights)
    
    
    dataset = {
        "observations": data['observations'].detach().cpu(),
        "actions": data['actions'].detach().cpu()
    }
    
    # rounds
    k = 5
    episodes_per_round = 5
    all_rewards = []
    for round in range(k):
        optimizer = optim.Adam(policy.parameters(), lr=lr)
        rewards = train(
            data=dataset, 
            optimizer=optimizer,
            device=device, 
            env=env,
            actor=policy,
            batch_size=batch_size, 
            eval_freq=eval_freq, 
            N=N,
            epochs=3,
        )
        all_rewards.append(rewards)
        print(f'round {round + 1} reward {rewards[-1]}')
        
        new_states = collect_states(policy, env, device, episodes_per_round)
        new_actions = collect_actions(expert, new_states, device)
        
        dataset['observations'] = np.concatenate([dataset['observations'], new_states], axis=0)
        dataset['actions'] = np.concatenate([dataset['actions'], new_actions], axis=0)
        
        # randomly shuffle
        id = np.random.permutation(len(dataset['observations']))
        dataset['observations'] = dataset['observations'][id]
        dataset['actions'] = dataset['actions'][id]
        
        
        
    plot(all_rewards[-1], eval_freq, batch_size, lr, N, True)
    