import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import gymnasium as gym

# python view_checkpoint.py

eval_path = 'eval/td3_continuous_action_step10000/outputs.npz'
cp_path = ''
data = np.load(eval_path)

obs = data['observations']
actions = data['actions']

print(f'shapes - obs: {obs.shape}, actions: {actions.shape}')

# view some obs, actions
for i in range(5):
    print(f'{i+1} - obs: {obs[i]}')
    print(f'action: {actions[i]}')
    print()

print()
print('obs')
print(f'min: {obs.min()}, max: {obs.max()}')
print(f'mean: {obs.mean(axis=0)}')

print()
print('actions')
print(f'min: {actions.min()}, max: {actions.max()}')
print(f'mean: {actions.mean(axis=0)}')

# plot it out
# path = 'periodic_saves/HalfCheetah-v4__td3_continuous_action__1__1749106273/td3_continuous_action_step10000'
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")


# plt.tricontour(obs, actions)
# plt.xlabel('observations')
# plt.ylabel('actions')
# plt.show()

# print(actions)





