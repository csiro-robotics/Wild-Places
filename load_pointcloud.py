import os 
import numpy as np 

def load_pointcloud(pointcloud_path, keep_intensity_channel=False):
    '''
    Load WildPlaces point cloud from disk
    Arguments:
    - pointcloud_path : Path to pointcloud to load
    - keep_intensity_channel : Whether or not to keep the intensity channel for the loaded pointcloud
    Returns:
    - Nx3/4 sized numpy array, where N is the number of points in the cloud and each point has 3 or 4 
      channels representing the x,y,z coordinates of the point and optionally the lidar intensity value
      for the point
    '''
    cloud = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1,4)
    if not keep_intensity_channel:
        cloud = cloud[:,:3]
    return cloud 