import os 
import pickle 
import torch 
import numpy as np 

def query_to_timestamp(query):
    base = os.path.basename(query)
    timestamp = float(base.replace('.pcd', ''))
    return timestamp

def euclidean_dist(query, database):
    return torch.cdist(torch.tensor(query).unsqueeze(0).unsqueeze(0), torch.tensor(database).unsqueeze(0)).squeeze().numpy()

def cosine_dist(query, database):
    return np.array(1 - torch.einsum('D,ND->N', torch.tensor(query), torch.tensor(database)))

def load_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        file = pickle.load(f)
    return file 

def get_latent_vectors(*args, **kwargs):
    # Placeholder function for user 
    raise NotImplementedError("No method for feature extraction currently implemented")