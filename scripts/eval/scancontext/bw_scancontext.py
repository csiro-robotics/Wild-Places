
from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import torch
import tqdm
from torchpack.utils.config import configs 
import open3d as o3d
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from sc_utils import *


def evaluate(log = False):
    # Run evaluation on all eval datasets
    log = False 
    show_progress = True

    eval_database_files = configs.eval.database_files 
    eval_query_files = configs.eval.query_files 

    print(eval_database_files)
    print(eval_query_files)

    assert len(eval_database_files) == len(eval_query_files)

    stats = {}
    mrr_grid = {}
    recall_1_grid = {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('/')[-1].split('_')[0]
        temp = query_file.split('/')[-1].split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        if os.path.exists(database_file):
            p = database_file
        else:
            p = os.path.join(configs.data.evalset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        if os.path.exists(query_file):
            p = query_file
        else:
            p = os.path.join(configs.data.evalset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp, mrr_temp, recall_1_temp = evaluate_dataset(database_sets, query_sets, log=log, show_progress=show_progress)
        stats[location_name] = temp
        mrr_grid[location_name] = mrr_temp
        recall_1_grid[location_name] = recall_1_temp

    return stats, mrr_grid, recall_1_grid


def evaluate_dataset(database_sets, query_sets, log: bool = False,
                     show_progress: bool = False):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    one_percent_recall = []
    mrr_list = []

    database_embeddings_sc, database_embeddings_rk = [], []
    query_embeddings_sc, query_embeddings_rk  = [], []

    for set in tqdm.tqdm(database_sets, disable=not show_progress, desc='Computing database embeddings'):
        scs, rks = get_latent_vectors(set)
        database_embeddings_sc.append(scs)
        database_embeddings_rk.append(rks)

    for set in tqdm.tqdm(query_sets, disable=not show_progress, desc='Computing query embeddings'):
        scs, rks = get_latent_vectors(set)
        query_embeddings_sc.append(scs)
        query_embeddings_rk.append(rks)

    pbar = tqdm.tqdm(total = len(query_sets)**2 - len(query_sets), desc = 'Eval')

    mrr_grid = np.identity(len(database_sets)) * 100
    recall_1_grid = np.identity(len(database_sets)) * 100

    fig, axes = plt.subplots(len(query_sets), len(query_sets))

    for i in range(len(query_sets)):
        for j in range(len(query_sets)):
            if i == j:
                continue
            pbar.update(1)
            pair_recall, pair_opr, mrr = get_recall(i, j, database_embeddings_sc, database_embeddings_rk, query_embeddings_sc, query_embeddings_rk, query_sets,
                                               database_sets, ax = axes[j][i], log=log) # Query is y axis, database is x axis
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            mrr_list.append(mrr)

            mrr_grid[i][j] = mrr
            recall_1_grid[i][j] = pair_recall[0]
            

    ave_recall = recall / count
    ave_one_percent_recall = np.mean(one_percent_recall)
    mean_recip_rank = np.mean(mrr_list)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall, 'mrr': mean_recip_rank}
    
    if configs.eval.save_grid_results != None:
        if not os.path.exists(configs.eval.save_grid_results):
            os.makedirs(configs.eval.save_grid_results)
        fig, ax = plt.subplots()
        ax = sns.heatmap(recall_1_grid.T, linewidth = 2.5, cmap = 'YlGn', annot = np.round(recall_1_grid.T, decimals = 1),
                        fmt = '', annot_kws={"size": 20 }, cbar = False, square=True, ax = ax)
        ax.set_xlabel('Database')
        ax.set_ylabel('Query')
        plt.savefig(os.path.join(configs.eval.save_grid_results, 'r1.png'), pad_inches = 0.1, dpi = 300)


    return stats, mrr_grid, recall_1_grid



@torch.no_grad()
def get_latent_vectors(t_set):
    # Adapted from original PointNetVLAD code
    
    scs, rks = [], []
    for fid in tqdm.tqdm(t_set):
        f = t_set[fid]['rel_scan_filepath']
        p = os.path.join(configs.data.dataset_folder, f)
        pcd = o3d.io.read_point_cloud(p)
        xyz = np.asarray(pcd.points)
        sc = get_sc(xyz)
        rk = sc2rk(sc)
        scs.append(sc)
        rks.append(rk)
    scs = np.asarray(scs)
    rks = np.asarray(rks)
    return scs, rks 


def get_recall(m, n, database_embeddings_sc, database_embeddings_rk, query_embeddings_sc, query_embeddings_rk, query_sets, database_sets, ax, log=False):
    # Original PointNetVLAD code
    database_output = database_embeddings_rk[m]
    queries_output = query_embeddings_rk[n]
    database_scs = database_embeddings_sc[m]
    query_scs = query_embeddings_sc[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors
    recall_idx = []

    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0

    # fig, ax = plt.subplots()

    database_coords = np.array([[database_sets[m][i]['easting'], database_sets[m][i]['northing']] for i in range(len(database_sets[m]))])
    ax.scatter(database_coords[:,0], database_coords[:,1], color = 'grey')
    coords = []
    colours = []

    for query_i in tqdm.tqdm(range(len(queries_output))):
        # i is query element ndx
        query_details = query_sets[n][query_i]    # {'query': path, 'northing': , 'easting': }
        coords.append([query_details['easting'], query_details['northing']])
        
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            colours.append('blue')
            continue
        num_evaluated += 1

        # Find nearest neightbours
        _, indices = database_nbrs.query(np.array([queries_output[query_i]]), k=num_neighbors)
        nn_ndx = indices[0]
        sc_dist = np.zeros((num_neighbors,))
        sc_yaw_diff = np.zeros((num_neighbors,))
        for nn_i in range(num_neighbors):
            candidate_ndx = nn_ndx[nn_i]
            candidate_sc = database_scs[candidate_ndx]
            query_sc = query_scs[query_i]
            sc_dist[nn_i], sc_yaw_diff[nn_i] = distance_sc(candidate_sc, query_sc)
        
        reranking_order = np.argsort(sc_dist)
        nn_ndx = nn_ndx[reranking_order]
        sc_yaw_diff = sc_yaw_diff[reranking_order]
        sc_dist = sc_dist[reranking_order]

        c = 'red'
        for j in range(len(nn_ndx)):
            if nn_ndx[j] in true_neighbors:
                recall[j] += 1
                recall_idx.append(j + 1)
                if j == 0:
                    c = 'green'
                break
        colours.append(c)

        
        

        if len(list(set(nn_ndx[0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    ax.set_title(f'{m} -> {n}')
    coords = np.array(coords)
    ax.scatter(x = coords[:,0], y = coords[:,1], c = colours)
    # plt.show()

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    mrr = np.mean(1 / np.array(recall_idx)) * 100

    return recall, one_percent_recall, mrr


def print_eval_stats(stats):
    stat_str_list = []
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        top1p = stats[database_name]['ave_one_percent_recall']
        mrr = stats[database_name]['mrr']
        recall_list = str(stats[database_name]['ave_recall'])
        t = f'Dataset: {database_name}\nAvg. top 1% recall: {top1p:.2f}   Avg. MMR: {mrr:.2f} Avg. recall @N:\n{recall_list}'
        print(t)
        # print(t.format(stats[database_name]['ave_one_percent_recall']))
        # print(stats[database_name]['ave_recall'])
        stat_str_list.append(t)
    save_str = '\n'.join(stat_str_list)
    return save_str


def pnv_write_eval_stats(file_name, prefix, stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in stats:
            ave_1p_recall = stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eval LoGG3D')
    parser.add_argument('--config', type=str, required=False, default='sc_eval_config.yaml')
    parser.add_argument('--save_grid_results', default = '/mnt/bracewell/git/egonn/third_party/scan_context/with_reranking_new_3m/')

    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive = True)
    configs.update(opts)
    configs.eval.save_grid_results = args.save_grid_results
    
    print('Training config path: {}'.format(args.config))
    save_dir = configs.eval.save_grid_results

    stats, mrr_grid, recall_1_grid = evaluate()

    save_str = print_eval_stats(stats)

    if save_dir != None:

        with open(os.path.join(configs.eval.save_grid_results, 'results.txt'), 'w') as f:
            f.write(save_str) 

        fig, axes = list(plt.subplots(1, len(mrr_grid)))
        fig.set_size_inches(5 * len(mrr_grid), 5)
        if len(mrr_grid) == 1:
            axes = [axes]
        else:
            axes = list(axes)

        for idx, k in enumerate(mrr_grid.keys()):
            ax = axes[idx]
            print(ax)
            mrr = mrr_grid[k]

            df_mrr = pd.DataFrame(columns=configs.eval.set_names)
            for idx, name in enumerate(configs.eval.set_names):
                df_mrr.loc[name] = mrr[idx]
            df_mrr.to_csv(os.path.join(save_dir, f'mrr_grid_{k}.csv'))

            ax = sns.heatmap(mrr, linewidth = 2.5, cmap = 'YlGn', annot = np.round(mrr, decimals = 1),
                        fmt = '', annot_kws={"size": 20 }, cbar = False, square=True, ax = ax)

            ax.set_title(f'MRR_{k}', fontsize=17)
            ax.set_xticklabels(configs.eval.set_names, fontsize=17)
            ax.set_yticklabels(configs.eval.set_names, fontsize=17)
            ax.set_xlabel('Database', fontsize=17)
            ax.set_ylabel('Query', fontsize=17)

        plt.savefig(os.path.join(save_dir, f'mrr.png'), bbox_inches = 'tight',
                pad_inches = 0.1, dpi = 300)
        
        # Recall@1
        fig, axes = plt.subplots(1, len(recall_1_grid))
        fig.set_size_inches(5 * len(mrr_grid), 5)
        if len(mrr_grid) == 1:
            axes = [axes]
        else:
            axes = list(axes)

        for idx, k in enumerate(recall_1_grid.keys()):
            ax = axes[idx]
            r1 = recall_1_grid[k]

            df_r1 = pd.DataFrame(columns=configs.eval.set_names)
            for idx, name in enumerate(configs.eval.set_names):
                df_r1.loc[name] = r1[idx]
            df_r1.to_csv(os.path.join(save_dir, f'r1_grid_{k}.csv'))

            ax = sns.heatmap(r1, linewidth = 2.5, cmap = 'YlGn', annot = np.round(r1, decimals = 1),
                        fmt = '', annot_kws={"size": 20 }, cbar = False, square=True, ax = ax)

            ax.set_title(f'r1_{k}', fontsize=17)
            ax.set_xticklabels(configs.eval.set_names, fontsize=17)
            ax.set_yticklabels(configs.eval.set_names, fontsize=17)
            ax.set_xlabel('Database', fontsize=17)
            ax.set_ylabel('Query', fontsize=17)

        plt.savefig(os.path.join(save_dir, f'r1.png'), bbox_inches = 'tight',
                pad_inches = 0.1, dpi = 300)

    