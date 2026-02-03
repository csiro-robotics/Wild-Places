import os

import numpy as np
import torch
from tqdm import tqdm

# Make sure to export python path to HOTFormerLoc repository before importing this script
from datasets.dataset_utils import create_batch
from datasets.augmentation import Normalize
from misc.utils import TrainingParams
from misc.torch_utils import to_device
from models.model_factory import model_factory


def load_params(config, model_config):
    params = TrainingParams(config, model_config)
    params.print()

    return params


def load_model(params):
    return model_factory(params.model_params)


def get_latent_vectors(model, data, params: TrainingParams):
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