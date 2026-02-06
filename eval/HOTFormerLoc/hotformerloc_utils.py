import os
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Make sure to export python path to HOTFormerLoc repository before importing this script
from datasets.dataset_utils import create_batch
from datasets.augmentation import Normalize
from misc.utils import TrainingParams
from misc.torch_utils import to_device
from models.model_factory import model_factory


class EvalDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        queries: Union[dict, list],
        transform=None,
        load_octree=False,
        coordinates="cartesian",
    ):
        assert os.path.exists(dataset_path), (
            "Cannot access dataset path: {}".format(dataset_path)
        )
        self.dataset_path = dataset_path
        if not isinstance(queries, (dict, list)):
            raise TypeError("Invalid query type, must be a dict with indices as keys or a list of queries")
        self.queries = queries
        self.transform = transform
        self.load_octree = load_octree
        self.coordinates = coordinates

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        query_info = self.queries[ndx]
        file_pathname = os.path.join(self.dataset_path, query_info['query'])
        data = load_pointcloud(file_pathname)
        if self.transform is not None:
            data = self.transform(data)
        if self.load_octree:
            # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
            mask = torch.all(abs(data) <= 1.0, dim=1)
            data = data[mask]
            # Also ensure this will hold if converting coordinate systems
            if self.coordinates == 'cylindrical':
                data_norm = torch.linalg.norm(data[:, :2], dim=1)[:, None]
                mask = torch.all(data_norm <= 1.0, dim=1)
                data = data[mask]
        return data


def make_eval_collate_fn(params: TrainingParams):
    """
    Custom collate function for evaluation dataloader. Only returns batches.
    """
    def collate_fn(data_list) -> dict:
        # Generate batches in correct format for HOTFormerLoc
        batch = create_batch(data_list, params.model_params.quantizer, params)
        return batch

    return collate_fn


def load_params(config, model_config):
    params = TrainingParams(config, model_config)
    params.print()

    return params


def load_model(params):
    return model_factory(params.model_params)


def get_latent_vectors(
    model: torch.nn.Module, data: Union[dict, list], params: TrainingParams
):
    """
    Efficient implementation of get_latent_vectors using PyTorch dataloaders.
    """
    # Initialise Dataset and Dataloader
    if params.normalize_points or params.scale_factor is not None:
        normalize_transform = Normalize(scale_factor=params.scale_factor,
                                        unit_sphere_norm=params.unit_sphere_norm)
    dataset = EvalDataset(
        dataset_path=params.dataset_folder,
        queries=data,
        transform=normalize_transform,
        load_octree=True,
        coordinates=params.model_params.coordinates,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=params.val_batch_size,
        shuffle=False,
        collate_fn=make_eval_collate_fn(params),
        num_workers=params.num_workers,
        pin_memory=True,
    )

    # Compute latent vectors for all queries
    vectors = []
    t = tqdm(desc="extracting latents", total=len(data))
    for batch in dataloader:
        batch = to_device(
            batch, device="cuda", non_blocking=True, construct_octree_neigh=True
        )
        
        # Compute global descriptor
        y = model(batch)
        vector = y['global'].detach().cpu()
        vectors.extend(vector)
        t.update(n=len(vector))
    t.close()
    vectors = torch.stack(vectors, dim=0)

    return vectors


def get_latent_vectors_OLD(model, data, params: TrainingParams):
    """
    Deprecated version -- highly inefficient.
    """
    if params.normalize_points or params.scale_factor is not None:
        normalize_transform = Normalize(scale_factor=params.scale_factor,
                                        unit_sphere_norm=params.unit_sphere_norm)
    vectors = []
    for query_info in tqdm(data.values(), "extracting latents", len(data.values())):
        fname = os.path.join(params.dataset_folder, query_info['query'])
        pc = load_pointcloud(fname)

        # Normalize points
        if params.normalize_points or params.scale_factor is not None:
            pc = normalize_transform(pc)
        # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
        mask = torch.all(abs(pc) <= 1.0, dim=1)
        pc = pc[mask]
        # Also ensure this will hold if converting coordinate systems
        if params.model_params.coordinates == 'cylindrical':
            pc_norm = torch.linalg.norm(pc[:, :2], dim=1)[:, None]
            mask = torch.all(pc_norm <= 1.0, dim=1)
            pc = pc[mask]

        # Create octree and move to GPU
        batch = create_batch([pc], params.model_params.quantizer, params)
        batch = to_device(batch, device="cuda", construct_octree_neigh=True)
        
        # Compute global descriptor
        y = model(batch)
        vector = y['global'].detach().cpu()
        vectors.append(vector)
    vectors = torch.cat(vectors, 0)

    return vectors

    
def load_pointcloud(fname):
    xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
    xyz = xyzr[:,:3]
    return torch.tensor(xyz)