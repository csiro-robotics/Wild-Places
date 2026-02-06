import os 
import numpy as np 
import pickle 
from glob import glob 

if __name__ == '__main__':
    pickles = glob('*test*.pickle')
    infos = [pickle.load(open(x, 'rb'))[1:] for x in pickles ]

    total_num = sum([len(x['filenames']) for x in infos[0]]) + sum([len(x['filenames']) for x in infos[1]])
    
    for idx in range(4):
        num_test = len(infos[0][idx]['filenames']) + len(infos[1][idx]['filenames'])
        num_train = total_num - num_test
        print(f"{num_train} & {num_test}")
    
    
    