import argparse 
import torch 
import os 
import numpy as np 
import pandas as pd 
from misc.utils import TrainingParams
from eval_wp.dataloader import get_descriptions_positions
from models.model_factory import model_factory
import pickle 
from tqdm import tqdm 
import faiss
import faiss.contrib.torch_utils

class Evaluator:
    def __init__(self, model, env_pickle_file, params, query_idx, debug=False):
        self.pos_thresh = 25.0 
        self.time_thresh = 600.0
        self.recall_values = list(range(1,26))
        self.params = params 
        
        env_data = pickle.load(open(env_pickle_file, 'rb'))
        self.environment = env_data.pop(0)
        self.env_info = env_data
        
        self.query_idx = query_idx 
        self.model = model 
        self.debug = debug 
        
    def get_recalls(self, info):
        name = info['name']
        features = info['feats']
        coords = info['coords']
        timestamps = info['timestamps']
        
        
        
        # Thresholds, other trackers
        thresholds = torch.linspace(0, 10, 1000)
        num_thresholds = len(thresholds)

        num_true_positive = torch.zeros(num_thresholds)
        num_false_positive = torch.zeros(num_thresholds)
        num_true_negative = torch.zeros(num_thresholds)
        num_false_negative = torch.zeros(num_thresholds)
        
        
        num_revisits = 0
        num_correct_loc = 0
        database_index_marker = 0
        time_start = timestamps[0]
        
        
        feat_dists = torch.cdist(features, features)
        position_dists = torch.cdist(coords, coords)
        recalls = np.zeros(len(self.recall_values))
        
        for q_idx in tqdm(range(len(features)), desc = f"Processing Sequence {name}"):
            # Get query info 
            q_feat = features[q_idx]
            q_coord = coords[q_idx]
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
                
            # Get top-1 candidate and distances in real world, embedding space 
            top1_idx = torch.argmin(dist_seen_embedding)
            _, topk_idx = torch.topk(dist_seen_embedding, len(self.recall_values), largest=False)
            
            
            for i, pidx in enumerate(topk_idx):
                if dist_seen_world[pidx] <= self.pos_thresh:
                    recalls[i:] += 1 
                    break 
                
        recalls = recalls / num_revisits * 100.0 
        return recalls 
        
    def run(self):
        env_feats_pos = [get_descriptions_positions(self.env_info[self.query_idx], self.model, self.params, self.debug)]
        
        df = pd.DataFrame(columns = [f"R@{n}" for n in self.recall_values])
        
        for seq_info in env_feats_pos:
            recalls = self.get_recalls(seq_info)
            df.loc[seq_info['name']] = recalls
            print(df.round(2))
        return self.environment, df.round(2) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HOTFormerLoc model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the model-specific configuration file')
    parser.add_argument('--debug', dest='debug', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--query_idx', type=int, required=True)
    parser.add_argument('--env_pickle_paths', type=str, nargs='+', required=True)
    args = parser.parse_args()
    
    assert os.path.exists(args.save_dir)
    
    # Make params 
    params = TrainingParams(args.config, args.model_config, debug=args.debug,
                            verbose=False)
    
    
    # Make model 
    model = model_factory(params.model_params)
    model = model.cuda()
    model.eval()
    
    # Load ckpt 
    ckpt = torch.load(args.checkpoint)['model_state_dict']
    model.load_state_dict(ckpt)
    
    for env_pickle_path in args.env_pickle_paths:
        environment, df_results = Evaluator(model, env_pickle_path, params, args.query_idx, args.debug).run()
        save_path = os.path.join(args.save_dir, f"{environment}_intra_qidx_{args.query_idx}.csv")
        df_results.to_csv(save_path)
