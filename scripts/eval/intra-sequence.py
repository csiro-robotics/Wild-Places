import os 
import argparse 
import pandas as pd 
import numpy as np 
from tqdm import tqdm 
from utils import get_latent_vectors, load_from_pickle, cosine_dist, euclidean_dist, query_to_timestamp

def eval_singlesession(database, embeddings, args):
    
    # Get embeddings, timestamps,coords and start time 
    database = load_from_pickle(database)

    if embeddings != None:
        embeddings = load_from_pickle(embeddings)
    else:
        embeddings = get_latent_vectors(database, embeddings)

    timestamps = [query_to_timestamp(database[k]['query']) for k in range(len(database.keys()))]
    coords = np.array([[database[k]['easting'], database[k]['northing']] for k in range(len(database.keys()))])
    start_time = timestamps[0]

    # Thresholds, other trackers
    thresholds = np.linspace(0, 1, 1000)
    num_thresholds = len(thresholds)

    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)

    # Get similarity function 
    if args.similarity_function == 'cosine':
        dist_func = cosine_dist
    elif args.similarity_function == 'euclidean':
        dist_func = euclidean_dist
    else:
        raise ValueError(f'No supported distance function for {args.similarity_function}')

    num_revisits = 0
    num_correct_loc = 0

    for query_idx in tqdm(range(len(database)), desc = 'Evaluating Embeddings'):
        q_embedding = embeddings[query_idx]
        q_timestamp = timestamps[query_idx]
        q_coord = coords[query_idx]

        # Exit if time elapsed since start is less than time threshold 
        
        if (q_timestamp - start_time - args.time_thresh) < 0:
            continue 

        # Build retrieval database 
        tt = next(x[0] for x in enumerate(timestamps) if x[1] > (q_timestamp - args.time_thresh))
        seen_embeddings = embeddings[:tt+1]
        seen_coords = coords[:tt+1]

        # Get distances in feature space and world 
        dist_seen_embedding = dist_func(q_embedding, seen_embeddings)
        dist_seen_world = euclidean_dist(q_coord, seen_coords)

        # Check if re-visit 
        if np.any(dist_seen_world < args.world_thresh):
            revisit = True 
            num_revisits += 1 
        else:
            revisit = False 

        # Get top-1 candidate and distances in real world, embedding space 
        top1_idx = np.argmin(dist_seen_embedding)
        top1_embed_dist = dist_seen_embedding[top1_idx]
        top1_world_dist = dist_seen_world[top1_idx]

        if top1_world_dist < args.world_thresh:
            num_correct_loc += 1 
        
        # Evaluate top-1 candidate 
        for thresh_idx in range(num_thresholds):
            threshold = thresholds[thresh_idx]

            if top1_embed_dist < threshold: # Positive Prediction
                if top1_world_dist < args.world_thresh:
                    num_true_positive[thresh_idx] += 1
                else:
                    num_false_positive[thresh_idx] += 1
            else: # Negative Prediction
                if not revisit:
                    num_true_negative[thresh_idx] += 1
                else:
                    num_false_negative[thresh_idx] += 1

    # Find F1Max and Recall@1 
    recall_1 = num_correct_loc / num_revisits

    F1max = 0.0 
    for thresh_idx in range(num_thresholds):
        nTruePositive = num_true_positive[thresh_idx]
        nFalsePositive = num_false_positive[thresh_idx]
        nTrueNegative = num_true_negative[thresh_idx]
        nFalseNegative = num_false_negative[thresh_idx]

        Precision = 0.0
        Recall = 0.0
        F1 = 0.0

        if nTruePositive > 0.0:
            Precision = nTruePositive / (nTruePositive + nFalsePositive)
            Recall = nTruePositive / (nTruePositive + nFalseNegative)
            F1 = 2 * Precision * Recall * (1/(Precision + Recall))

        if F1 > F1max:
            F1max = F1 

    return {'F1max': F1max, 'Recall@1': recall_1, 'Sequence Length': len(embeddings), 'Num. Revisits': num_revisits, 'Num. Correct Locations': num_correct_loc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading and saving paths 
    parser.add_argument('--databases', required = True, type = str, nargs = '+', help = 'List of paths to pickles containing info about database sets')
    parser.add_argument('--database_features', default = None, type = str, nargs = '+', help = 'List of paths to pickles containing feature vectors for database sets')
    parser.add_argument('--run_names', type = str, nargs = '+', help = 'List of names of runs being evaluated')
    parser.add_argument('--save_dir', type = str, default = None, help = 'Save Directory for results csv')
    # Eval parameters
    parser.add_argument('--world_thresh', type = float, default = 5, help = 'Distance to be considered revisit in world')
    parser.add_argument('--time_thresh', type = float, default = 300, help = 'Time before a previous frame can be considered a valid revisit')
    parser.add_argument('--similarity_function', type = str, default = 'cosine', help = 'Distance function used to calculate similarity of embeddings')
    args = parser.parse_args()

    stats = pd.DataFrame(columns = ['F1max', 'Recall@1', 'Sequence Length', 'Num. Revisits', 'Num. Correct Locations'])
    for database, embeddings, location in zip(args.databases, args.database_features, args.run_names):
        temp_stats = eval_singlesession(database, embeddings, args)
        stats.loc[location] = [temp_stats['F1max'], temp_stats['Recall@1'], temp_stats['Sequence Length'], temp_stats['Num. Revisits'], temp_stats['Num. Correct Locations']]
    print(stats)

    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        stats.to_csv(os.path.join(args.save_dir, 'intra-run_results.csv'), index = False)


