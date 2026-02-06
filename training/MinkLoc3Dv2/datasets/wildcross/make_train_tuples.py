import os 
import numpy as np 
import pandas as pd 
from sklearn.neighbors import KDTree 
import pickle 
import argparse 
from tqdm import tqdm 
from scipy.spatial.transform import Rotation as R

from datasets.base_datasets import TrainingTuple


_OFFSET = 2000


def construct_query_dict(df_centroids, filepaths, save_path, ind_nn_r, ind_r_r):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    # Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
    # Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    tree = KDTree(df_centroids[['x', 'y']])
    ind_nn = tree.query_radius(df_centroids[['x', 'y']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['x', 'y']], r=ind_r_r)
    queries = {}
    for anchor_ndx in tqdm(range(len(ind_nn))):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['x', 'y']])
        pose = np.array(df_centroids.iloc[anchor_ndx][['x','y','z','qx','qy','qz','qw']])
        query = filepaths[anchor_ndx]
        
        # Extract timestamp from the filename
        timestamp = float(os.path.basename(query).replace('.bin', ''))
        
        # scan_filename = os.path.split(query)[1]
        # timestamp = float(os.path.splitext(scan_filename)[0])

        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        tt = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position = anchor_pos)
        queries[anchor_ndx] = tt

    file_path = save_path
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type = str, required = True, help = 'Dataset root folder')
    parser.add_argument('--save_folder', type = str, required = True, help = 'Folder to save training pickles to')
    parser.add_argument('--pos_thresh', type = float, default = 3, help  = 'Threshold to sample positives within')
    parser.add_argument('--neg_thresh', type = float, default = 20, help = 'Threshold to sample negative within')
    args = parser.parse_args()
    
    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    print(f'Dataset root: {args.dataset_root}')

    args.save_folder = args.dataset_root if args.save_folder is None else args.save_folder
    print(f'Saving pickles to: {args.save_folder}')
    
    os.makedirs(args.save_folder, exist_ok=True)

    for split_idx in [0,1,2,3]:    
        # Venman 
        venman_seqs = ['V-01','V-02','V-03','V-04']
        venman_seqs.pop(split_idx)
        df_venman = pd.concat([
            pd.read_csv(os.path.join(args.dataset_root, seq, 'submap_poses.csv'))
            for seq in venman_seqs], axis=0
        )
        venman_filepaths = []
        for seq in venman_seqs:
            seq_filepaths = [os.path.join(seq, 'Clouds_downsampled',x ) 
                            for x in sorted(os.listdir(os.path.join(args.dataset_root, seq, 'Clouds_downsampled')))]
            venman_filepaths += seq_filepaths
        assert len(venman_filepaths) == len(df_venman)
        
        # Karawatha 
        karawatha_seqs = ['K-01','K-02','K-03','K-04']
        karawatha_seqs.pop(split_idx)
        df_karawatha = pd.concat([
            pd.read_csv(os.path.join(args.dataset_root, seq, 'submap_poses.csv'))
            for seq in karawatha_seqs], axis=0
        )
        df_karawatha[['x','y']] += _OFFSET
        karawatha_filepaths = []
        for seq in karawatha_seqs:
            seq_filepaths = [os.path.join(seq, 'Clouds_downsampled',x ) 
                            for x in sorted(os.listdir(os.path.join(args.dataset_root, seq, 'Clouds_downsampled')))]
            karawatha_filepaths += seq_filepaths
        assert len(karawatha_filepaths) == len(df_karawatha)
        
        
        df_train = pd.concat([df_venman, df_karawatha], axis=0)
        filepaths_train = venman_filepaths + karawatha_filepaths
        
        print(f"Venman: {df_venman} Training Submaps")
        print(f"Karawatha: {df_karawatha} Training Submaps")
        print(f"Total: {df_train} Training Submaps")
        
        construct_query_dict(df_train, filepaths_train,
                            os.path.join(args.save_folder, f"training_wildcross_split_idx{split_idx}.pickle"),
                            ind_nn_r=args.pos_thresh, ind_r_r=args.neg_thresh)
        