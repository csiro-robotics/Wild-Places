import pandas as pd 
import numpy as np 
from glob import glob 
from scipy.spatial.transform import Rotation as R
import torch 
import os 
import yaml 
from shapely.geometry import Polygon, Point
import torchvision.transforms as T
import PIL.Image as Image 
import pickle 
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm 
import faiss
import faiss.contrib.torch_utils
import ocnn
from ocnn.octree import Octree, Points

from datasets.WildCross.wildcross_train import WildCrossPointCloudLoader
from datasets.CSWildPlaces.CSWildPlaces_train import ValTransform as CSWildPlacesValTransform
from datasets.augmentation import Normalize
from datasets.coordinate_utils import CylindricalCoordinates


class EvalDataset:
    def __init__(self, sequence_info, params):
        self.filenames = sequence_info['filenames']
        self.pc_loader = WildCrossPointCloudLoader()
        self.params = params 
        self.normalize_transform = Normalize(   scale_factor=params.scale_factor,
                                                unit_sphere_norm=params.unit_sphere_norm)
        self.coord_converter = CylindricalCoordinates(use_octree=True)
        
        
        # self.transform = transform 
        # self.coordinates = coordinates
        # self.load_octree = load_octree
        # self.octree_depth = octree_depth
        # self.full_depth = full_depth
        # self.quantizer = quantizer

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = self.filenames[idx]
        data = self.pc_loader(file_path)
        data = torch.tensor(data)
        if self.params.normalize_points or self.params.scale_factor is not None:
            data = self.normalize_transform(data)
        
        if self.params.load_octree:  # Convert to Octree format
            # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
            mask = torch.all(abs(data) <= 1.0, dim=1)
            data = data[mask]
            # Also ensure this will hold if converting coordinate systems
            if self.params.model_params.coordinates == 'cylindrical':
                data_norm = torch.linalg.norm(data[:, :2], dim=1)[:, None]
                mask = torch.all(data_norm <= 1.0, dim=1)
                data = data[mask]
                # Convert to cylindrical coords
                data = self.coord_converter(data)
            # Convert to ocnn Points object, then create Octree
            points = Points(data)
            data = Octree(self.params.octree_depth, full_depth=2, batch_size=1)
            data.build_octree(points)            
        
        return data 

def collate_fn(data_list):
    octrees = ocnn.octree.merge_octrees(data_list)
    batch = {'octree': octrees}
    return batch

def make_dataloader(sequence_info, params):
    dataset = EvalDataset(sequence_info, params)
    
    dataloader = DataLoader(
        dataset,
        batch_size=16, 
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn
    )
    return dataloader 


@torch.no_grad()
def get_descriptions_positions(sequence_info, model, params, debug=False):
    dataloader = make_dataloader(sequence_info, params)
    seq_feats = []
    name = sequence_info['name']
    if debug:
        seq_feats = torch.randn(len(dataloader.dataset), 256)
    else:
        for batch in tqdm(dataloader, desc = f"Extracting features for sequence {name}", total = len(dataloader)):
            batch = {k:v.cuda() for k,v in batch.items()}
            batch['octree'].construct_all_neigh()
            feats = model.forward(batch)['global']
            seq_feats.append(feats.cpu())
    
        seq_feats = torch.cat(seq_feats)

    seq_pos = torch.tensor(sequence_info['coords'])
    seq_ts = torch.tensor(sequence_info['timestamps'])
    
    
    seq_processed_data = {'name': sequence_info['name'], 'feats': seq_feats, 'coords': seq_pos, 'timestamps': seq_ts}
    return seq_processed_data
        


