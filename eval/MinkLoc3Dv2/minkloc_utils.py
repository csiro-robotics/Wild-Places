import os

import MinkowskiEngine as ME
import numpy as np
import torch

# Make sure to export python path to MinkLoc3Dv2 repository before importing this script
from misc.utils import TrainingParams
from models.model_factory import model_factory


def load_params(config, model_config):
    params = TrainingParams(config, model_config)
    params.print()

    return params


def load_model(params):
    return model_factory(params.model_params)


def get_latent_vectors(model, data, params: TrainingParams):
    vectors = []
    for query_info in data.values():
        fname = os.path.join(params.dataset_folder, query_info['query'])
        pc = load_pointcloud(fname)

        coords, _ = params.model_params.quantizer(pc)
        bcoords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        batch = {'coords': bcoords.to('cuda'), 'features': feats.to('cuda')}

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