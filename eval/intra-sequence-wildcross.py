import os 
import pickle 
import argparse 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KDTree
import faiss 

pd.options.display.float_format = "{:,.2f}".format


import sys 
import torch 

def model_factory(*args, **kwargs):
    """
    Placeholder function for model_factory
    This function should return the model for whatever LPR model is being evaluated,
    in order for the descriptors to be extracted for a given pre-trained checkpoint 
    """
    return None 

class Evaluator:
    def __init__(self, model, pickle_file, split_idx, dataset_root, save_dir, debug=False):
        self.pos_thresh = 3.0 
        self.time_thresh = 600.0
        self.recall_values = list(range(1,26))


        self.model = model 
        self.env_data = pickle.load(open(pickle_file, 'rb'))
        self.split_idx = split_idx 
        self.dataset_root = dataset_root 
        self.save_dir = save_dir
        self.debug = debug 
        
    def get_recalls(self, seq_data):
        name = seq_data['name']
        features = seq_data['feats']
        coords = seq_data['coords']
        timestamps = seq_data['timestamps']
    
        num_revisits = 0
        time_start = timestamps[0]
        
        feat_dists = torch.cdist(features, features)
        position_dists = torch.cdist(coords, coords)
        recalls = np.zeros(len(self.recall_values))
        
        for q_idx in tqdm(range(len(features)), desc = f"Processing Sequence {name}"):
            # Get query info 
            q_timestamp = timestamps[q_idx]
            
            # Skip if time elapsed since start is less than the time threshold
            if (q_timestamp - time_start - self.time_thresh) < 0:
                continue 
            
            # Build retrieval database 
            tt = next(x[0] for x in enumerate(timestamps) if x[1] > (q_timestamp - self.time_thresh))
            dist_seen_embedding = feat_dists[q_idx, :tt+1]
            dist_seen_world = position_dists[q_idx, :tt+1]
            
            # Check if re-visit 
            if torch.any(dist_seen_world < self.pos_thresh) and len(dist_seen_embedding) > 25:
                num_revisits += 1 
            else:
                continue 
                
            # Get distances in real world, embedding space 
            _, topk_idx = torch.topk(dist_seen_embedding, len(self.recall_values), largest=False)
            
            for i, pidx in enumerate(topk_idx):
                if dist_seen_world[pidx] <= self.pos_thresh:
                    recalls[i:] += 1 
                    break 
                
        recalls = recalls / num_revisits * 100.0 
        return recalls 

    @torch.no_grad()
    def get_latent_vectors(self, *args, **kwargs):
        """
        Placeholder function for get_latent_vectors
        This function should take any number of arguments, and return the descriptors for each point cloud in a given sequence as 
        an NxD pytorch float tensor
        """
        return None
    
    def get_positions(self, seq_data):
        positions = []
        for i in range(len(seq_data)):
            positions.append([seq_data[i]['easting'], seq_data[i]['northing']])
        positions = torch.tensor(positions).float()
        return positions 


    def get_descriptors_positions(self):
        seq_data = self.env_data[self.split_idx]
        seq_name = seq_data[0]['seq_name']
        descriptors = self.get_latent_vectors(seq_data, seq_name, debug=self.debug)
        positions = self.get_positions(seq_data)
        seq_timestamps = torch.tensor([v['timestamp'] for v in seq_data.values()])

        info = {
            'name': seq_name,
            'feats': descriptors,
            'coords': positions,
            'timestamps': seq_timestamps
        }

        return info 

    def run(self):
        seq_data = self.get_descriptors_positions()
        recalls = self.get_recalls(seq_data)
        df_results = pd.DataFrame(columns = [f"R@{n}" for n in self.recall_values])
        df_results.loc[seq_data['name']] = recalls
        df_results = df_results.round(2)
        return df_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pickle_files', type=str, nargs='+', required=True)
    parser.add_argument('--location_names', type=str, nargs = '+', default=['Venman','Karawatha'])
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--split_idx', type=int, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    model = model_factory() # Placeholder function
    ckpt = torch.load(args.ckpt)['model_state_dict']
    model.load_state_dict(ckpt)
    model.to('cuda')
    model.eval()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    assert len(args.test_pickle_files) == len(args.location_names)

    for pickle_file, location_name in zip(args.test_pickle_files, args.location_names):
        evaluator = Evaluator(model, pickle_file, args.split_idx, args.dataset_root, args.save_dir, args.debug)
        df_env_results = evaluator.run()
        print(df_env_results)

        if args.save_dir:
            df_env_results.to_csv(os.path.join(args.save_dir, f"{location_name}_intrasequence_results_split_{args.split_idx}.csv"))