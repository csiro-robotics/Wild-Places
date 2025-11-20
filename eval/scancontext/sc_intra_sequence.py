from tqdm import tqdm 
import torch 
import numpy as np 
import argparse 
import os 
import pickle 
import pandas as pd
from torchpack.utils.config import configs 
import open3d as o3d
import numpy_indexed as npi
from sklearn.neighbors import KDTree

from sc_utils import *


@torch.no_grad()
def get_latent_vectors(t_set):
    # Adapted from original PointNetVLAD code
    
    scs, rks = [], []
    for fid in tqdm(t_set):
        f = t_set[fid]['query']
        p = os.path.join(configs.data.dataset_folder, f)
        xyz = np.fromfile(p, dtype=np.float32).reshape(-1,4)[:,:3]
        sc = get_sc(xyz)
        rk = sc2rk(sc)
        scs.append(sc)
        rks.append(rk)
    scs = np.asarray(scs)
    rks = np.asarray(rks)
    return scs, rks 


def evaluate_single_run():
    # Run evaluation on all eval datasets
    stats = pd.DataFrame(columns = ['F1max', 'R1', 'Sequence Length', 'Num. Revisits', 'Num. Correct Loc'])
    pickles_venman = pickle.load(open(configs.eval.database_files[0], 'rb'))
    pickles_karawatha = pickle.load(open(configs.eval.database_files[1], 'rb'))

    target_seq = {
        'VEN-03': pickles_venman[2],
        'VEN-04': pickles_venman[3],
        'KAR-03': pickles_karawatha[2],
        'KAR-04': pickles_karawatha[3],
    }

    for name, database_set in target_seq.items():
        F1max, R1, seq_len, num_revisits, num_correct_loc = get_single_run_stats(database_set, name)
        stats.loc[name] = [F1max, R1, seq_len, num_revisits, num_correct_loc]
        stats = stats.round(2)
        print(stats)
    return stats 

def euclidean_distance(query, database):
    return torch.cdist(torch.tensor(query).unsqueeze(0).unsqueeze(0), torch.tensor(database).unsqueeze(0)).squeeze().numpy()      

def query_to_timestamp(query):
    base = os.path.basename(query)
    timestamp = float(base.replace('.bin', ''))
    return timestamp

def get_single_run_stats(database_set, run_name, embeddings_sk = [], embeddings_rk = []):
    if len(embeddings_sk) == 0:
        embeddings_sk, embeddings_rk = get_latent_vectors(database_set) # N x D, in chronological order
    timestamps = [query_to_timestamp(database_set[k]['query']) for k in range(len(database_set.keys()))]
    coords = np.array([[database_set[k]['easting'],database_set[k]['northing']] for k in range(len(database_set.keys()))])
    start_time = timestamps[0]

    world_thresh = configs.eval.world_thresh 
    time_thresh = configs.eval.time_thresh

    # Thresholds, other trackers
    thresholds = np.linspace(0.001, 1.0, 1000) # TODO Make this a variable from thresh_min, thresh_max, num_thresholds
    num_thresholds = len(thresholds)

    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)
        
    num_revisits = 0
    num_correct_loc = 0

    for query_idx in tqdm(range(len(database_set)), desc = 'Evaluating Embeddings'):

        query_embedding = embeddings_rk[query_idx]
        query_timestamp = timestamps[query_idx]
        query_coord = coords[query_idx]

        # Sanity check time 
        if (query_timestamp - start_time - time_thresh) < 0:
            continue 
        
        # Build retrieval database 
        tt = next(x[0] for x in enumerate(timestamps) if x[1] > (query_timestamp - time_thresh))
        seen_embeddings = embeddings_rk[:tt+1] # Seen x D
        seen_coords = coords[:tt+1] # Seen x 2

        # Get distances in feat space and real world
        dist_seen_world = euclidean_distance(query_coord, seen_coords)

        database_nbrs = KDTree(seen_embeddings)
        num_neighbors = min(25, len(seen_embeddings))
        # Find nearest neightbours
        _, indices = database_nbrs.query(np.array([query_embedding]), k=num_neighbors)
        nn_ndx = indices[0]
        sc_dist = np.zeros((num_neighbors,))
        sc_yaw_diff = np.zeros((num_neighbors,))
        for nn_i in range(num_neighbors):
            candidate_ndx = nn_ndx[nn_i]
            candidate_sc = embeddings_sk[candidate_ndx]
            query_sc = embeddings_sk[query_idx]
            sc_dist[nn_i], sc_yaw_diff[nn_i] = distance_sc(candidate_sc, query_sc)
        
        reranking_order = np.argsort(sc_dist)
        nn_ndx = nn_ndx[reranking_order]
        sc_yaw_diff = sc_yaw_diff[reranking_order]
        sc_dist = sc_dist[reranking_order]

        # Check if re-visit
        if np.any(dist_seen_world < world_thresh):
            revisit = True 
            num_revisits += 1
        else:
            revisit = False 

        # Get top-1 candidate and distances in real world, embedding space 
        top1_idx = nn_ndx[0]
        top1_embed_dist = sc_dist[0]#dist_seen_embedding[top1_idx]
        top1_world_dist = dist_seen_world[top1_idx]

        if top1_world_dist < world_thresh:
            num_correct_loc += 1

        # Evaluate top-1 candidate 
        for thresh_idx in range(num_thresholds):
            threshold = thresholds[thresh_idx]

            if top1_embed_dist < threshold: # Positive Prediction
                if top1_world_dist < world_thresh:
                    num_true_positive[thresh_idx] += 1
                else: #elif top1_world_dist > 20:
                    num_false_positive[thresh_idx] += 1
            else: # Negative Prediction
                if not revisit:
                    num_true_negative[thresh_idx] += 1
                else:
                    num_false_negative[thresh_idx] += 1

    # Find F1Max and Recall@1 
    if num_revisits == 0:
        recall_1 = np.nan 
    else:
        recall_1 = num_correct_loc / num_revisits

    F1max = 0.0 
    for thresh_idx in range(num_thresholds):
        nTruePositive = num_true_positive[thresh_idx]
        nFalsePositive = num_false_positive[thresh_idx]
        nTrueNegative = num_true_negative[thresh_idx]
        nFalseNegative = num_false_negative[thresh_idx]

        nTotalTestPlaces = nTruePositive + nFalsePositive + nTrueNegative + nFalseNegative

        Precision = 0.0
        Recall = 0.0
        Prev_Recall = 0.0
        F1 = 0.0

        if nTruePositive > 0.0:
            Precision = nTruePositive / (nTruePositive + nFalsePositive)
            Recall = nTruePositive / (nTruePositive + nFalseNegative)
            F1 = 2 * Precision * Recall * (1/(Precision + Recall))

        if F1 > F1max:
            F1max = F1 
            thresh_max = thresholds[thresh_idx]

    print(f'Num Revisits : {num_revisits}')
    print(f'Num. Correct Locations : {num_correct_loc}')
    print(f'Sequence Length : {len(database_set)}')
    print(f'Recall@1: {recall_1}')
    print(f'F1max: {F1max}')


    return F1max * 100, recall_1 * 100, len(database_set), num_revisits, num_correct_loc
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evluate ScanContext model')
    parser.add_argument('--config', type=str, required=False, default='sc_eval_config.yaml')
    parser.add_argument('--save_dir', type = str, default = None)
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive = True)
    configs.update(opts)
    
    print('Training config path: {}'.format(args.config))

    # Evaluate 
    stats = evaluate_single_run()
    print(stats)

