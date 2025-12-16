# Warsaw University of Technology

import numpy as np
from typing import List, Sequence
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KDTree
import ocnn
from ocnn.octree import Octree, Points

from datasets.base_datasets import EvaluationTuple, TrainingDataset
from datasets.augmentation import TrainSetTransform
from datasets.pointnetvlad.pnv_train import PNVTrainingDataset
from datasets.pointnetvlad.pnv_train import TrainTransform as PNVTrainTransform
from datasets.pointnetvlad.pnv_train import ValTransform as PNVValTransform
from datasets.CSWildPlaces.CSWildPlaces_train import CSWildPlacesTrainingDataset
from datasets.CSWildPlaces.CSWildPlaces_train import TrainTransform as CSWildPlacesTrainTransform
from datasets.CSWildPlaces.CSWildPlaces_train import ValTransform as CSWildPlacesValTransform
from datasets.samplers import BatchSampler
from misc.utils import TrainingParams
from datasets.base_datasets import PointCloudLoader
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader
from datasets.CSWildPlaces.CSWildPlaces_raw import CSWildPlacesPointCloudLoader
from datasets.WildCross.wildcross_train import WildCrossTrainingDataset

def get_pointcloud_loader(dataset_type) -> PointCloudLoader:
    if 'CSWildPlaces' in dataset_type or 'WildPlaces' in dataset_type:
        return CSWildPlacesPointCloudLoader
    else:
        return PNVPointCloudLoader()


def make_datasets(params: TrainingParams, validation: bool = True):
    # Create training and validation datasets
    datasets = {}
    train_set_transform = TrainSetTransform(params.set_aug_mode, random_rot_theta=params.random_rot_theta)

    if params.dataset_name == 'WildCross':
        train_transform = CSWildPlacesTrainTransform(params.aug_mode, normalize_points=params.normalize_points,
                                                     scale_factor=params.scale_factor, unit_sphere_norm=params.unit_sphere_norm,
                                                     zero_mean=params.zero_mean, random_rot_theta=params.random_rot_theta)
        datasets['train'] = WildCrossTrainingDataset(params.dataset_folder, params.train_file,
                                                      transform=train_transform, set_transform=train_set_transform,
                                                      load_octree=params.load_octree, octree_depth=params.octree_depth,
                                                      full_depth=params.full_depth, coordinates=params.model_params.coordinates)



    elif 'CSWildPlaces' in params.dataset_name or 'WildPlaces' in params.dataset_name:
        train_transform = CSWildPlacesTrainTransform(params.aug_mode, normalize_points=params.normalize_points,
                                                     scale_factor=params.scale_factor, unit_sphere_norm=params.unit_sphere_norm,
                                                     zero_mean=params.zero_mean, random_rot_theta=params.random_rot_theta)
        datasets['train'] = CSWildPlacesTrainingDataset(params.dataset_folder, params.train_file,
                                                      transform=train_transform, set_transform=train_set_transform,
                                                      load_octree=params.load_octree, octree_depth=params.octree_depth,
                                                      full_depth=params.full_depth, coordinates=params.model_params.coordinates)
        if validation:
            val_transform = CSWildPlacesValTransform(normalize_points=params.normalize_points, scale_factor=params.scale_factor,
                                                     unit_sphere_norm=params.unit_sphere_norm, zero_mean=params.zero_mean)
            datasets['val'] = CSWildPlacesTrainingDataset(params.dataset_folder, params.val_file,
                                                        transform=val_transform,
                                                        load_octree=params.load_octree, octree_depth=params.octree_depth,
                                                        full_depth=params.full_depth, coordinates=params.model_params.coordinates)
    # PoinNetVLAD datasets (RobotCar and Inhouse)
    # PNV datasets have their own transform
    else:  # used for Oxford and CS-Campus3D
        train_transform = PNVTrainTransform(params.aug_mode, normalize_points=params.normalize_points,
                                            scale_factor=params.scale_factor, unit_sphere_norm=params.unit_sphere_norm,
                                            zero_mean=params.zero_mean, random_rot_theta=params.random_rot_theta)
        datasets['train'] = PNVTrainingDataset(params.dataset_folder, params.train_file,
                                               transform=train_transform, set_transform=train_set_transform,
                                               load_octree=params.load_octree, octree_depth=params.octree_depth,
                                               full_depth=params.full_depth, coordinates=params.model_params.coordinates)
        if validation:
            val_transform = PNVValTransform(normalize_points=params.normalize_points, scale_factor=params.scale_factor,
                                            unit_sphere_norm=params.unit_sphere_norm, zero_mean=params.zero_mean)
            datasets['val'] = PNVTrainingDataset(params.dataset_folder, params.val_file,
                                                 transform=val_transform,
                                                 load_octree=params.load_octree, octree_depth=params.octree_depth,
                                                 full_depth=params.full_depth, coordinates=params.model_params.coordinates)

    return datasets


def create_batch(clouds: Sequence[torch.Tensor], quantizer, params: TrainingParams):
    """
    Util function to create batches in correct format from an input list of 
    point clouds.

    Args:
        clouds (Sequence[Tensor]): Sequence of point clouds of shape (N, 3).
        quantizer (Optional): If using MinkLoc, quantizer for sparse quantization.
            If using OctFormer, coordinate system converter.
        params (TrainingParams): Training parameters for the model.
    """
    if quantizer is not None:
        clouds = [quantizer(e) for e in clouds]
    octrees = []
    # Convert to ocnn Points object, then create Octree
    for cloud in clouds:
        cloud_points_obj = Points(cloud)
        octree = Octree(params.octree_depth, params.full_depth)
        octree.build_octree(cloud_points_obj)
        octrees.append(octree)
    octrees_merged = ocnn.octree.merge_octrees(octrees)
    # NOTE: remember to construct the neighbor indices before processing (much faster on GPU)
    # octrees_merged.construct_all_neigh()
    batch = {'octree': octrees_merged}
    return batch


def make_collate_fn(dataset: TrainingDataset, quantizer, params: TrainingParams):
    # quantizer: converts to polar (when polar coords are used) and quantizes
    # batch_split_size: if not None, splits the batch into a list of multiple mini-batches with batch_split_size elems
    # octree: if True, loads octree in batch instead of sparse tensor
    def collate_fn(data_list):
        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]
        
        # clouds = data
        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            lens = [len(cloud) for cloud in clouds]
            clouds_merged = torch.cat(clouds, dim=0)
            clouds_merged = dataset.set_transform(clouds_merged)
            clouds = clouds_merged.split(lens)

        # Compute positives and negatives mask
        # dataset.queries[label]['positives'] is bitarray
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Generate batches in correct format for OctFormer
        if params.batch_split_size is None or params.batch_split_size == 0:
            batch = create_batch(clouds, quantizer, params)
        else:
            # Split the batch into chunks
            batch = []
            for i in range(0, len(clouds), params.batch_split_size):
                temp = clouds[i:i + params.batch_split_size]
                minibatch = create_batch(temp, quantizer, params)
                batch.append(minibatch)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and negatives_mask which are
        # batch_size x batch_size boolean tensors
        #return batch, positives_mask, negatives_mask, torch.tensor(sampled_positive_ndx), torch.tensor(relative_poses)
        return batch, positives_mask, negatives_mask

    return collate_fn


def make_dataloaders(params: TrainingParams, validation=True):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params, validation=validation)

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    quantizer = params.model_params.quantizer
    train_collate_fn = make_collate_fn(datasets['train'], quantizer, params)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler,
                                     collate_fn=train_collate_fn, num_workers=params.num_workers,
                                     pin_memory=True)
    if validation and 'val' in datasets:
        val_collate_fn = make_collate_fn(datasets['val'], quantizer, params)
        val_sampler = BatchSampler(datasets['val'], batch_size=params.val_batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)

    return dataloders


def filter_query_elements(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple],
                          dist_threshold: float) -> List[EvaluationTuple]:
    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    map_pos = np.zeros((len(map_set), 2), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        map_pos[ndx] = e.position

    # Build a kdtree
    kdtree = KDTree(map_pos)

    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(query_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1

    print(f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] "
          f"radius")
    return filtered_query_set


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e

