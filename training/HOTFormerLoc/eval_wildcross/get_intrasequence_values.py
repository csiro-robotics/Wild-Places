import os 
import pandas as pd 
import argparse 
import numpy as np 
from glob import glob 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    args = parser.parse_args()
    
    df_paths = sorted([x for x in glob(os.path.join(args.root_dir, "V*intra*csv"))]) + sorted([x for x in glob(os.path.join(args.root_dir, "K*intra*csv"))])
    df_results = pd.concat([pd.read_csv(dfp, index_col=0) for dfp in df_paths])
    assert len(df_results) == 8 
    df_results.loc['Average'] = df_results.mean(0)
    print(df_results[['R@1','R@5']].round(2))
    