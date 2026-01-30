# Some functions copied from here: https://github.com/csiro-robotics/LoGG3D-Net/blob/main/models/pipelines/pipeline_utils.py
# Some functions copied from here: https://github.com/chrischoy/FCGF/blob/master/util/pointcloud.py
# load_pointcloud() adapted from here: https://github.com/csiro-robotics/LoGG3D-Net/blob/main/utils/data_loaders/general/general_sparse_dataset.py

import os
import sys

import numpy as np
import open3d as o3d
import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate
from tqdm import tqdm

# Make sure to export python path to LoGG3D-Net repository before importing this script
from models.pipeline_factory import get_pipeline


def load_model():
    return get_pipeline('LOGG3D')


def get_latent_vectors(model, data, dataset_root):
    vectors = []
    for query_info in tqdm(data.values(), "extracting latents", len(data.values())):
        fname = os.path.join(dataset_root, query_info['query'])
        sp_tensor = load_pointcloud(fname)
        sp_tensor = sp_tensor.to('cuda')
        y = model(sp_tensor)
        gds = y[0].detach().cpu()
        gds = gds.reshape((-1,1024))
        vectors.append(gds)
    vectors = torch.cat(vectors, 0)
    return vectors


def make_open3d_point_cloud(xyz, color=None, tile=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if tile:
            if len(color) != len(xyz):
                color = np.tile(color, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def load_pointcloud(fname):
    xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
    pcd = make_open3d_point_cloud(xyzr[:, :3], color=None)
    xyz = np.asarray(pcd.points)
    oo = np.ones(len(xyz)).reshape((-1,1))
    xyzr = np.hstack((xyz, oo)).astype(np.float32)

    return make_sparse_tensor(xyzr, 0.5)


def make_sparse_tensor(lidar_pc, voxel_size=0.05, return_points=False):
    # get rounded coordinates
    coords = np.round(lidar_pc[:, :3] / voxel_size)
    coords -= coords.min(0, keepdims=1)
    feats = lidar_pc

    # sparse quantization: filter out duplicate points
    _, indices = sparse_quantize(coords, return_index=True)
    coords = coords[indices]
    feats = feats[indices]

    # construct the sparse tensor
    inputs = SparseTensor(feats, coords)
    inputs = sparse_collate([inputs])
    inputs.C = inputs.C.int()
    if return_points:
        return inputs, feats
    else:
        return inputs