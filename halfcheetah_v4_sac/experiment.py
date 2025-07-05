import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import gymnasium as gym
import torch.optim as optim
import time

# network implementation
from actor_impl import Actor, DiffusionActor, OGActor, NewDiffusionActor
from helper_functions import batch, run_eval, run_eval_gauss, run_eval_diff, run_eval_diff_vec, run_eval_pretrain
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from diffusers import DDPMScheduler

env_id = "HalfCheetah-v4"


# plots the reward over time
def plot(rewards, eval_freq, batch_size, lr, N, name=None, save_plot=False):
    x = np.arange(len(rewards)) * eval_freq
    plt.figure()
    plt.plot(x, rewards)
    plt.xlabel("training batches")
    plt.ylabel("rewards")
    plt.title(f'bs {batch_size}, lr {lr}, eval_freq {eval_freq}, N {N}')
    plt.grid(True)
    if save_plot:
        os.makedirs('halfcheetah_v4_sac/plots', exist_ok=True)
        if name is not None:
            fname = f'halfcheetah_v4_sac/plots/{name}.png'
        else:
            fname = f'halfcheetah_v4_sac/plots/bs{batch_size}_lr{lr}_ef{eval_freq}_N{N}.png'
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()
    print(f'plotted bs {batch_size}, lr {lr}, N {N}')

# old train function
def train(data, weights, device, batch_size, lr, eval_freq, N, weight_decay, epochs=3):
    start_time = time.time()
    print("started training")
    # init
    env = gym.make(env_id)
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    actor = Actor(env).to(device)
    
    new_sd = actor.state_dict()

    D = actor.action_dim
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
    for i in range(epochs):
        epoch_time = time.time()
        for j, b in enumerate(batched_data):
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
            if j == 0:
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
            if j % eval_freq == 0:
                # print(f'eval batch {i}')
                eval_time = time.time()
                actor.eval()
                rewards.append(run_eval_gauss(actor, env, device, N))
                actor.train()
                print(f"Eval {j} time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - eval_time))}")
                
        print(f"Epoch {i} time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_time))}")
    
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print(f'final reward: {rewards[-1]}')
    
    return rewards 


def train_diffusion(data, weights, device, batch_size, lr, eval_freq, N, 
                    weight_decay, scheduler, pretrain_lr = 1e-4, pretrain_epochs=3, 
                    epochs=3, T=25, num_env=5):
    start_time = time.time()
    print("started training")
    # init
    env = gym.make(env_id)
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    
    actor = DiffusionActor(env, scheduler=scheduler, T=T).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=lr, weight_decay=weight_decay)
    batched_data = batch(data, batch_size=batch_size)
    
    
    # pretraining
    target_time = time.time()
    target_actor = OGActor(env).to(device)
    target_actor.load_state_dict(weights)
    
    print(f'Target actor baseline: {run_eval(target_actor, env, device, N)}')
    target_optimizer = optim.Adam(actor.parameters(), lr=pretrain_lr, weight_decay=weight_decay)
    print("entering pretraining")
    total_reward = 0
    for i in range(pretrain_epochs):
        epoch_time = time.time()
        for j, b in enumerate(batched_data):
            states = torch.stack([s for s, _ in b]).float().to(device=device)
            with torch.no_grad():
                action, log_prob, mean = target_actor.get_action(states)
            
            if i < 3:
                t_rand = torch.zeros(states.size(0), dtype=torch.long, device=device)
            else:
                t_rand = torch.randint(0, T, (states.size(0),), device=device)
            
            # alpha_b = actor.alpha_bar[t_rand].unsqueeze(1)
            noise = torch.randn_like(mean, device=device)
            # action_noise = torch.sqrt(alpha_b) * mean + torch.sqrt(1 - alpha_b) * noise
            action_noise = scheduler.add_noise(mean, noise, t_rand)
            
            pred = actor(states, action_noise, t_rand)
            loss = F.mse_loss(pred, noise)
            total_reward += loss.item() * states.size(0)
            
            target_optimizer.zero_grad()
            loss.backward()
            target_optimizer.step()
            
        print(f"Epoch {i} time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_time))}")
    
    print(f"Target time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - target_time))}")
    print(f"avg reward: {total_reward/len(data['observations'])}")
    print("post-pretrain reward: ", run_eval_diff_vec(actor, env_id, device, N, num_env=num_env))
    
    
    
    # training loop
    actor.train()
    rewards = []
    print("entering training loop")
    for i in range(epochs):
        epoch_time = time.time()
        for j, b in enumerate(batched_data):
            states = torch.stack([s for s, _ in b]).float().to(device=device)
            actions = torch.stack([a for _, a in b]).float().to(device=device)
            
            t = torch.randint(0, actor.T, (states.size(0),), device=device)
            # alpha_bar_t = actor.alpha_bar[t].unsqueeze(1)
            noise = torch.randn_like(actions)
            # action_noise = torch.sqrt(alpha_bar_t) * actions + torch.sqrt(1 - alpha_bar_t) * noise
            action_noise = scheduler.add_noise(actions, noise, t)
            
            pred = actor(states, action_noise, t)
            loss = F.mse_loss(pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # eval
            if j % eval_freq == 0:
                # print(f'eval batch {i}')
                eval_time = time.time()
                # actor.eval()
                rewards.append(run_eval_diff_vec(actor, env_id, device, N, num_env=num_env))
                # actor.train()
                print(f"Eval {j} time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - eval_time))}")
                
        print(f"Epoch {i} time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_time))}")
    
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print(f'final reward: {rewards[-1]}')
    
    return rewards 


def new_train_diffusion(data, weights, device, batch_size, lr, eval_freq, N, 
                    weight_decay, scheduler, pretrain_lr = 1e-4, pretrain_epochs=3, 
                    epochs=3, T=25, num_env=5):
    start_time = time.time()
    print("started training")
    # init
    env = gym.make(env_id)
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    
    actor = NewDiffusionActor(env, scheduler, T=T).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=lr, weight_decay=weight_decay)
    batched_data = batch(data, batch_size=batch_size)
    
    # pretraining
    print("entering pretraining")
    target_time = time.time()
    target_optimizer = optim.Adam(actor.pretrain_net.parameters(), lr=pretrain_lr, weight_decay=weight_decay)
    
    # warm starting actor with weights
    og = OGActor(env).to(device)
    og.load_state_dict(weights)
    new_sd = actor.state_dict()
    
    print("OGActor.fc1: ", og.fc1.weight.shape,
        "pretrain_net[0]: ", actor.pretrain_net[0].weight.shape)
    print("OGActor.fc_mean: ", og.fc_mean.weight.shape,
        "pretrain_net[2]: ", actor.pretrain_net[2].weight.shape)

    w_before = actor.pretrain_net[0].weight.data.view(-1)[0].item()
    print("pretrain_net[0].weight[0] before copy:", w_before)
    
    # manually add to prevent differences
    for k, v in og.state_dict().items():
        if k == "fc1.weight" and new_sd["pretrain_net.0.weight"].shape == v.shape:
            new_sd["pretrain_net.0.weight"] = v.clone()
        if k == "fc1.bias" and new_sd["pretrain_net.0.bias"].shape == v.shape:
            new_sd["pretrain_net.0.bias"] = v.clone()
        if k == "fc_mean.weight" and new_sd["pretrain_net.2.weight"].shape == v.shape:
            new_sd["pretrain_net.2.weight"] = v.clone()
        if k == "fc_mean.bias" and new_sd["pretrain_net.2.bias"].shape == v.shape:
            new_sd["pretrain_net.2.bias"] = v.clone()
    
    actor.load_state_dict(new_sd)
    
    w_after = actor.pretrain_net[0].weight.data.view(-1)[0].item()
    print("pretrain_net[0].weight[0] after copy: ", w_after)
    print("changed", w_before != w_after)
    

    for i in range(pretrain_epochs):
        epoch_time = time.time()
        epoch_loss = 0
        for j, b in enumerate(batched_data):
            states = torch.stack([s for s, _ in b]).float().to(device=device)
            actions = torch.stack([a for _, a in b]).float().to(device=device)
            
            pred = actor.predict(states)
            loss = F.mse_loss(pred, actions)
            
            target_optimizer.zero_grad()
            loss.backward()
            target_optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(data['observations']) / batch_size)
        print(f'Pretrain epoch {i + 1}/{pretrain_epochs} MSE loss: {avg_loss}')
        print(f"Epoch {i + 1} time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_time))}")
    
    print(f"Target time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - target_time))}")
    # print("post-pretrain reward: ", run_eval_diff_vec(actor, env_id, device, N, num_env=num_env))
    print("post-pretrain reward: ", run_eval_pretrain(actor, env, device, N))
    
    
    
    # training loop
    actor.train()
    rewards = []
    print("entering training loop")
    for i in range(epochs):
        epoch_time = time.time()
        epoch_loss = 0
        for j, b in enumerate(batched_data):
            states = torch.stack([s for s, _ in b]).float().to(device=device)
            actions = torch.stack([a for _, a in b]).float().to(device=device)
            
            t = torch.randint(0, actor.T, (states.size(0),), device=device)
            noise = torch.randn_like(actions)
            action_noise = scheduler.add_noise(actions, noise, t)
            
            pred = actor(states, action_noise, t)
            loss = F.mse_loss(pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # eval
            if j % eval_freq == 0:
                eval_time = time.time()
                # actor.eval()
                rewards.append(run_eval_diff_vec(actor, env_id, device, N, num_env=num_env))
                # actor.train()
                print(f"Eval {j} time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - eval_time))}")
        
        avg_loss = epoch_loss / (len(data['observations']) / batch_size)
        print(f'Epoch {i + 1} MSE loss: {avg_loss}')
        print(f"Epoch {i + 1}/{epochs} time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_time))}")
    
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print(f'final reward: {rewards[-1]}')
    
    return rewards 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load("halfcheetah_v4_sac/halfcheetah_v4_data.pt", map_location=device, weights_only=False)
    weights = torch.load("halfcheetah_v4_sac/halfcheetah_v4_actor_weights.pt", map_location=device)
    
    # hparams
    batch_size = 500
    lr = 5e-5
    eval_freq = 100
    N = 30
    pretrain_lr = 5e-4
    pretrain_epochs=30
    epochs = 1
    weight_decay = 1e-6
    T = 100
    num_env=min(N, 16) # 16 is num of my CPU cores
    
    scheduler = DDPMScheduler(
        num_train_timesteps=T,
        beta_schedule="linear",
        # clip_sample=False
    )
    scheduler.set_timesteps(T)
    scheduler.timesteps = scheduler.timesteps.to(device)
    
    rewards = new_train_diffusion(
        data=data, 
        weights=weights, 
        device=device, 
        batch_size=batch_size, 
        lr=lr, 
        eval_freq=eval_freq, 
        N=N, 
        weight_decay=weight_decay,
        scheduler=scheduler,
        pretrain_lr=pretrain_lr,
        pretrain_epochs=pretrain_epochs,
        epochs=epochs,
        T=T, 
        num_env=num_env
    )
    plot(rewards, eval_freq, batch_size, lr, N, "new_run", True)
    
    # batch_size = [500, 1000, 2000, 4000]
    # lr = [1e-3, 1e-4, 5e-4]
    # eval_freq = 10
    # N = 10
    # epochs = 1
    
    # for bs in batch_size:
    #     for l in lr:
    #         rewards = train(data, weights, device, bs, l, eval_freq, N, epochs=epochs)
    #         plot(rewards, eval_freq, bs, l, N, True)
    
