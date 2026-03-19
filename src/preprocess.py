import open3d as o3d
import numpy as np


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.2):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downsampled.points)