import os 
import numpy as np 
from datasets.base_datasets import PointCloudLoader

class WildPlacesPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None
        
    def read_pc(self, file_pathname):
        pc = np.fromfile(file_pathname, dtype=np.float32).reshape(-1,4)[:,:3]
        return pc 