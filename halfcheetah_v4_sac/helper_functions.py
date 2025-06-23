import numpy as np

# returns num_split amount of batches
# in each batch, this will return a list of (s, a) elements
def batch(data, batch_size=500):
    num_splits = len(data['observations']) // batch_size
    
    obs = np.array_split(data['observations'], num_splits)
    actions = np.array_split(data['actions'], num_splits)
    
    return [list(zip(o, a)) for o, a in zip(obs, actions)]