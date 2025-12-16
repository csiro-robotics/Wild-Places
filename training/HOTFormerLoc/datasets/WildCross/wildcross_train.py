# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# For information on dataset see: https://github.com/mikacuy/pointnetvlad
# Warsaw University of Technology

import numpy as np
from datasets.base_datasets import TrainingDataset
from datasets.CSWildPlaces.CSWildPlaces_raw import CSWildPlacesPointCloudLoader

import os 
from datasets.base_datasets import PointCloudLoader


class WildCrossPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 ndarray
        assert os.path.splitext(file_pathname)[-1] == ".bin"
        pc = np.fromfile(file_pathname, dtype=np.float32).reshape(-1,4)[:,:3]
        
        return pc

class WildCrossTrainingDataset(TrainingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_loader = WildCrossPointCloudLoader()
        
