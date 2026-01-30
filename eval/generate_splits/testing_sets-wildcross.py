import os 
import pickle 
import argparse 
import pandas as pd 
from tqdm import tqdm 
from sklearn.neighbors import KDTree 

def load_csv(csv_path, rel_cloud_path):
    df = pd.read_csv(csv_path, delimiter = ',', dtype = str)
    df = df.astype({'x': float, 'y':float, 'z':float, 'qx':float, 'qy':float, 'qz':float, 'qw':float, '%time': str})
    df['easting'] = df['x']
    df['northing'] = df['y']
    # df = df.rename(columns = {'x': 'easting', 'y': 'northing'})
    df['filename'] = rel_cloud_path + '/' + df['%time'] + '.bin'
    df = df[['%time', 'filename', 'northing', 'easting', 'x','y','z','qx','qy','qz','qw']]
    print(df['filename'])
    return df

def construct_testing_pickles(dataset_root, folders):

    all_sequences_info = []

    for folder in tqdm(folders):
        df = load_csv(os.path.join(dataset_root, folder, 'submap_poses.csv'), os.path.join(folder, 'Clouds_downsampled'))
        info = {}
        for idx, row in df.iterrows():
            info[len(info)] = {'seq_name': folder,
                               'query': row['filename'], 
                               'northing': row['northing'], 
                               'easting': row['easting'], 
                               'pose': row[['x','y','z','qx','qy','qz','qw']],
                               'timestamp': float(row['%time'])}
        all_sequences_info.append(info)

    return all_sequences_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='Root folder for WildCross dataset')
    parser.add_argument('--save_folder', type=str, required=True, help='Path to save pickle files to ')
    args = parser.parse_args()

    venman_info = construct_testing_pickles(args.dataset_root, ['V-01','V-02','V-03','V-04'])
    with open(os.path.join(args.save_folder, "venman_testing_info.pickle"), 'wb') as f:
        pickle.dump(venman_info, f)

    karawatha_info = construct_testing_pickles(args.dataset_root, ['K-01','K-02','K-03','K-04'])
    with open(os.path.join(args.save_folder, "karawatha_testing_info.pickle"), 'wb') as f:
        pickle.dump(karawatha_info, f)