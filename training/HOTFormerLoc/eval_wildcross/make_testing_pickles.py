import os 
import pickle 
import numpy as np 
import pandas as pd 
from glob import glob 
import argparse 

class MakeEvalPickles:
    def __init__(self, root_folder, save_folder):
        self.root_folder = root_folder 
        
        sequences_venman = ['V-01', 'V-02', 'V-03', 'V-04']
        sequences_karawatha = ['K-01', 'K-02', 'K-03', 'K-04']
        
        info_venman = ['Venman']
        info_karawatha = ['Karawatha']
        
        for idx, seqv in enumerate(sequences_venman):
            paths, coords = self.get_paths_coords(seqv)
            info_seq = {}
            info_seq['name'] = seqv
            info_seq['filenames'] = paths 
            info_seq['coords'] = coords 
            info_seq['timestamps'] = np.float64([os.path.basename(x).strip('.bin') for x in paths])
            info_venman.append(info_seq)
            
        for seqk in sequences_karawatha:
            paths, coords = self.get_paths_coords(seqk)
            info_seq = {}
            info_seq['name'] = seqk
            info_seq['filenames'] = paths 
            info_seq['coords'] = coords 
            info_seq['timestamps'] = np.float64([os.path.basename(x).strip('.bin') for x in paths])
            info_karawatha.append(info_seq) 
            
        with open(os.path.join(save_folder, "venman_test_info_crossfold.pickle"), 'wb') as f:
            pickle.dump(info_venman, f)
        with open(os.path.join(save_folder, "karawatha_test_info_crossfold.pickle"), 'wb') as f:
            pickle.dump(info_karawatha, f) 
            
    def get_paths_coords(self, sequence):
        # Get images 
        filepaths = sorted(glob(os.path.join(self.root_folder, sequence, 'Clouds_downsampled', '*.bin')))
        
        # Get coords 
        df = pd.read_csv(os.path.join(self.root_folder, sequence, 'submap_poses_aligned.csv'))
        coords = df[['x','y','z']].to_numpy()
        
        return filepaths, coords

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder')
    parser.add_argument('--save_folder')
    args = parser.parse_args()
    
    os.makedirs(args.save_folder, exist_ok=True)
    
    MakeEvalPickles(args.root_folder, args.save_folder)
