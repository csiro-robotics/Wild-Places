import os 
import pandas as pd 
import argparse 
import numpy as np 
from glob import glob 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    args = parser.parse_args()
    
    df_results = pd.DataFrame(columns=["R@1","R@5"])
    
    # Venman 
    df_paths = [x for x in glob(os.path.join(args.root_dir, "Venman_*csv")) if 'intra' not in x]
    assert len(df_paths) == 4
    recall_1_list = []
    recall_5_list = []
    for dfp in df_paths:
        df = pd.read_csv(dfp).iloc[:3]
        recall_1_list.append(df["R@1"].to_numpy())
        recall_5_list.append(df['R@5'].to_numpy())
    assert len(np.concatenate(recall_1_list)) == 12 
    assert len(np.concatenate(recall_5_list)) == 12 
    r1 = round(np.concatenate(recall_1_list).mean(),2)
    r5 = round(np.concatenate(recall_5_list).mean(),2)
    df_results.loc['Venman'] = [r1,r5]
    
    # Karawatha 
    df_paths = [x for x in glob(os.path.join(args.root_dir, "Karawatha_*csv")) if 'intra' not in x]
    assert len(df_paths) == 4
    recall_1_list = []
    recall_5_list = []
    for dfp in df_paths:
        df = pd.read_csv(dfp).iloc[:3]
        recall_1_list.append(df["R@1"].to_numpy())
        recall_5_list.append(df['R@5'].to_numpy())
    assert len(np.concatenate(recall_1_list)) == 12 
    assert len(np.concatenate(recall_5_list)) == 12 
    r1 = round(np.concatenate(recall_1_list).mean(),2)
    r5 = round(np.concatenate(recall_5_list).mean(),2)
    df_results.loc['Karawatha'] = [r1,r5]
    
    df_results.loc['Average'] = df_results.mean(axis=0).round(2)
    print(df_results)
