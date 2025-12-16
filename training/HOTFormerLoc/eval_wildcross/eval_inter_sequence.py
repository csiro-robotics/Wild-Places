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
    def __init__(self, model, env_pickle_path, params, query_idx, debug):
        self.model = model 
        env_data = pickle.load(open(env_pickle_path, 'rb'))
        self.environment = env_data.pop(0)
        self.env_info = env_data
        self.params = params 
        self.query_idx = query_idx 
        self.debug = debug 
        self.recall_values = [1,5]
        self.pos_thresh = 3.0
        
    def get_positives(self, q_pos, db_pos):
        cross_dist = torch.cdist(q_pos, db_pos)
        valid_queries = []
        valid_positives = []
        for idx, row in enumerate(cross_dist):
            positives = torch.nonzero(row <= self.pos_thresh).flatten().numpy()
            if len(positives) == 0:
                continue 
            valid_queries.append(idx)
            valid_positives.append(positives)
            
        return valid_queries, valid_positives
        
    def get_recalls(self, query_feat_pos, db_feat_pos):
        q_name = query_feat_pos['name']
        q_feats = query_feat_pos['feats']
        q_coords = query_feat_pos['coords']
        db_name = db_feat_pos['name']
        db_feats = db_feat_pos['feats']
        db_coords = db_feat_pos['coords']
        
        
        # res = faiss.StandardGpuResources()
        db_faiss_index = faiss.IndexFlatL2(db_feats.shape[1])
        # db_faiss_index = faiss.index_cpu_to_gpu(res, 0, db_faiss_index)
        db_faiss_index.add(db_feats)
        
        valid_query_idx, positive_idx = self.get_positives(q_coords, db_coords)
        print(f"{q_name}->{db_name}: {len(q_coords) - len(valid_query_idx)} removed due to no valid positives")
        
        q_feats = torch.index_select(q_feats, 0, torch.tensor(valid_query_idx))
        q_coords = torch.index_select(q_coords, 0, torch.tensor(valid_query_idx))
        
        _, predictions = db_faiss_index.search(q_feats, max(self.recall_values))
        # Get recalls 
        recalls = np.zeros(len(self.recall_values))
        for q_idx, preds in enumerate(tqdm(predictions, desc = f"{q_name}->{db_name}")):
            for i, n in enumerate(self.recall_values):
                if np.any(np.isin(preds[:n], positive_idx[q_idx])):
                    recalls[i:] += 1 
                    break 
        
        # Divide by number of queries and multiply by 100 
        recalls = recalls / len(q_feats) * 100
        print(f"{q_name}->{db_name} : {recalls.round(2)}")
        return recalls 
        
    def run(self):
        query_info = self.env_info.pop(self.query_idx)
        query_feat_pos = get_descriptions_positions(query_info, self.model, self.params, self.debug)
        db_data = [get_descriptions_positions(db_info, self.model, self.params, self.debug) for db_info in self.env_info]
        
        df_results = pd.DataFrame(columns = [f"R@{n}" for n in self.recall_values])
        for db_feat_pos in db_data:
            recalls = self.get_recalls(query_feat_pos, db_feat_pos)
            df_results.loc[db_feat_pos['name']] = recalls 
            
        df_results = df_results.round(2)
        print(df_results)
        return self.environment, df_results

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
        save_path = os.path.join(args.save_dir, f"{environment}_inter_qidx_{args.query_idx}.csv")
        df_results.to_csv(save_path)
    
