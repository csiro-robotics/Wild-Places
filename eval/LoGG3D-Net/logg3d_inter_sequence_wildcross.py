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

import logg3d_utils

def model_factory():
    return logg3d_utils.load_model()

class Evaluator:
    def __init__(self, model, pickle_file, split_idx, dataset_root, save_dir, debug=False):
        self.pos_thresh = 3.0 
        self.recall_values = list(range(1,26))


        self.model = model 
        self.env_data = pickle.load(open(pickle_file, 'rb'))
        self.split_idx = split_idx 
        self.dataset_root = dataset_root 
        self.save_dir = save_dir
        self.debug = debug 
        
    def get_positives(self, q_coords, db_coords):
        ''' 
        Given the positions of each image in the query and database sets, return 
        the ids of the positive matches for each query 
        '''
        # Using MM slightly affects the number of positive query-database pairs
        cross_dist = torch.cdist(q_coords, db_coords, 2, 'donot_use_mm_for_euclid_dist')
        valid_queries = []
        valid_positives = []
        for idx, row in enumerate(cross_dist):
            positives = torch.nonzero(row <= self.pos_thresh).flatten()
            if len(positives) == 0:
                continue 
            valid_queries.append(idx)
            valid_positives.append(positives)
            
        return valid_queries, valid_positives
    
    def get_recalls(self, query_data, db_data):
        q_name = query_data['name']
        q_feats = query_data['feats']
        q_coords = query_data['coords']
        db_name = db_data['name']
        db_feats = db_data['feats']
        db_coords = db_data['coords']
        
        db_faiss_index = faiss.IndexFlatL2(db_feats.shape[1])
        db_faiss_index.add(db_feats)
        
        q_valid_idx, q_positive_idx = self.get_positives(q_coords, db_coords)
        print(f"{q_name}->{db_name}: {len(q_coords) - len(q_valid_idx)} queries removed due to no valid positives")
        
        # Remove queries, coords for examples with no valid positives before getting predictions
        q_feats = torch.index_select(q_feats, 0, torch.tensor(q_valid_idx))
        q_coords = torch.index_select(q_coords, 0, torch.tensor(q_valid_idx))
        _, predictions = db_faiss_index.search(q_feats, max(self.recall_values))
        
        # Get recalls 
        recalls = np.zeros(len(self.recall_values))
        for q_idx, preds in enumerate(tqdm(predictions, desc = f"{q_name}->{db_name}")):
            q_positives = q_positive_idx[q_idx]
            for i, n in enumerate(self.recall_values):
                if np.any(np.isin(preds[:n], q_positives)):
                    recalls[i:] += 1 
                    break 
                    
        # Divide by number of queries and multiply by 100 
        recalls = recalls / len(q_feats) * 100
        return recalls

    @torch.no_grad()
    def get_latent_vectors(self, env_data, seq_name, debug):
        vectors = logg3d_utils.get_latent_vectors(self.model, env_data, self.dataset_root)
        
        # Save latent vectors if debug is True and save dir has been specified
        if debug and self.save_dir:
            save_path = os.path.join(self.save_dir, f'{seq_name}_vectors_split_{args.split_idx}.pickle')
            with open(save_path, 'wb') as handle:
                pickle.dump(vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return vectors
    
    def get_positions(self, seq_data):
        positions = []
        for i in range(len(seq_data)):
            positions.append([seq_data[i]['easting'], seq_data[i]['northing']])
        positions = torch.tensor(positions).float()
        return positions 


    def get_descriptors_positions(self):
        all_sequence_info = []
        for idx in range(len(self.env_data)):
            seq_name = self.env_data[idx][0]['seq_name']
            descriptors = self.get_latent_vectors(self.env_data[idx], seq_name, debug=self.debug)
            positions = self.get_positions(self.env_data[idx])

            all_sequence_info.append({
                'name': seq_name,
                'feats': descriptors,
                'coords': positions
            })

        return all_sequence_info

    def run(self):


        all_sequences_data = self.get_descriptors_positions()
        query_data = all_sequences_data.pop(self.split_idx)
        df_results = pd.DataFrame(columns = [f"R@{n}" for n in self.recall_values])
        
        for db_data in all_sequences_data:
            recalls = self.get_recalls(query_data, db_data)
            df_results.loc[db_data['name']] = recalls 
            
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

    model = model_factory()
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
            df_env_results.to_csv(os.path.join(args.save_dir, f"{location_name}_intersequence_results_split_{args.split_idx}.csv"))