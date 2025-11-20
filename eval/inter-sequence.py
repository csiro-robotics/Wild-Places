import os 
import pickle 
import argparse 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KDTree

from utils import get_latent_vectors, load_from_pickle
pd.options.display.float_format = "{:,.2f}".format

def evaluate(queries, databases, query_features, database_features, args):
    assert len(queries) == len(databases), f'Length of Query and Database Dictionaries is not the same: {len(queries)} vs {len(databases)}'

    stats = pd.DataFrame(columns = ['Recall@1', 'Recall@1%', 'MRR'])
    
    for query_sets, database_sets, query_feat, database_feat, location in zip(queries, databases, query_features, database_features, args.location_names):
        
        # Extract features if not provided
        if query_feat == None or database_feat == None:
            query_feat, database_feat = get_latent_vectors(query_sets, database_sets)
        
        # Load run information, feature vectors 
        query_sets = load_from_pickle(query_sets)
        database_sets = load_from_pickle(database_sets)
        query_feat = load_from_pickle(query_feat)
        database_feat = load_from_pickle(database_feat)

        temp_stats = evaluate_location(query_sets, database_sets, query_feat, database_feat, location, args)
        stats.loc[location] = [temp_stats['ave_recall'][0], temp_stats['ave_one_percent_recall'], temp_stats['mrr']]

    stats.loc['Average'] = stats.mean(axis = 0)
    return stats


def evaluate_location(query_sets, database_sets, query_feat, database_feat, location, args):
    # Run evaluation on a single location 
    recall = np.zeros(25)
    count = 0 
    recall_1p = []
    mrr = []

    mrr_grid = np.identity(len(database_sets)) * 100
    recall_1_grid = np.identity(len(database_sets)) * 100

    pbar = tqdm(total = len(query_sets)**2 - len(query_sets), desc = f'Eval: {location}')

    for i in range(len(query_sets)):
        for j in range(len(query_sets)):
            if i == j:
                continue 
            pbar.update(1)
            pair_recall, pair_recall_1p, pair_mrr = get_recall(i, j, database_feat, query_feat, database_sets, query_sets)

            recall += np.array(pair_recall)
            recall_1p.append(pair_recall_1p)
            mrr.append(pair_mrr)
            count += 1 

            recall_1_grid[i][j] = pair_recall[0]
            mrr_grid[i][j] = pair_mrr 

    ave_recall = recall / count 
    ave_recall_1p = np.mean(recall_1p)
    ave_mrr = np.mean(mrr)

    
    stats = {'ave_recall': ave_recall, 'ave_one_percent_recall': ave_recall_1p, 'mrr': ave_mrr}

    if args.save_grid_results and args.save_dir != None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        fig, (ax1, ax2) = plt.subplots(1,2)
        
        # Recall@1
        ax1 = sns.heatmap(recall_1_grid.T, linewidth = 2.5, cmap = 'YlGn', annot = np.round(recall_1_grid.T, decimals = 1),
                          fmt = '', annot_kws={"size": 20 }, cbar = False, square=True, ax = ax1)
        ax1.set_xlabel('Database')
        ax1.set_ylabel('Query')

        # MRR
        ax2 = sns.heatmap(mrr_grid.T, linewidth = 2.5, cmap = 'YlGn', annot = np.round(mrr_grid.T, decimals = 1),
                          fmt = '', annot_kws={"size": 20 }, cbar = False, square=True, ax = ax2)
        ax2.set_xlabel('Database')
        ax2.set_ylabel('Query')

        plt.savefig(os.path.join(args.save_dir, f"results_grid_{location}.png"), pad_inches = 0.1, dpi = 300)

    return stats



def get_recall(m, n, database_feat, query_feat, database_sets, query_sets):

    database_feat_run = database_feat[m]
    queries_feat_run = query_feat[n]

    # Get database info 
    database_nbrs = KDTree(database_feat_run)

    # Set up variables 
    num_neighbours = 25 
    recall = np.zeros(num_neighbours)
    recall_idx = []

    one_percent_retrieved = 0 
    threshold = max(int(round(len(database_feat_run) / 100)), 1)

    num_evaluated = 0 

    
    
    
    for i in range(len(queries_feat_run)):
        query_details = query_sets[n][i]
        true_neighbours = query_details[m]
        if len(true_neighbours) == 0:
            continue 
        num_evaluated += 1 

        # Find nearest neighbours 
        _, indices = database_nbrs.query(np.asarray([queries_feat_run[i]]), k = num_neighbours)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbours:
                recall[j] += 1 
                recall_idx.append(j + 1)
                break 

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbours)))) > 0:
            one_percent_retrieved += 1

    recall = np.cumsum(recall) / float(num_evaluated) * 100
    recall_1p = one_percent_retrieved / float(num_evaluated) * 100 
    mrr = np.mean(1 / np.array(recall_idx)) * 100 

    return recall, recall_1p, mrr 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=str, required = True, nargs = '+', help = 'List of paths to pickles containing info about query sets')
    parser.add_argument('--databases', type = str, required = True, nargs = '+',  help = 'List of paths to pickles containing info about query sets')
    parser.add_argument('--query_features', type = str, default = None, nargs = '+', help = 'List of paths to pickles containing feature vectors for query sets')
    parser.add_argument('--database_features', type = str, default = None, nargs = '+', help = 'List of paths to pickles containing feature vectors for database sets')
    parser.add_argument('--location_names', type = str, nargs = '+', default = ['Venman', 'Karawatha'], help = 'Names of the locations for each respective query & database')
    parser.add_argument('--save_dir', type = str, default = None, help = 'Save Directory for results csv')
    parser.add_argument('--save_grid_results',  default = False, action = 'store_true', help = 'Flag for saving inter-run recall as a heatmap')

    args = parser.parse_args()

    

    stats = evaluate(args.queries, args.databases, args.query_features, args.database_features, args)
    print(stats)

    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        stats.to_csv(os.path.join(args.save_dir, 'inter-run_results.csv'), index = False)
    

