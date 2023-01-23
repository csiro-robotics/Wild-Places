from pyntcloud import PyntCloud

def read_pc(file_pathname):
    # Reads the point cloud without pre-processing
    # Returns Nx3 ndarray
    pointcloud = PyntCloud.from_file(file_pathname)
    pc = np.array(pointcloud.points)[:,:3]
    return pc 