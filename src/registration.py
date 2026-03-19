import open3d as o3d
import numpy as np


def run_icp(source_points: np.ndarray, target_points: np.ndarray, threshold: float = 1.5):
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    source.points = o3d.utility.Vector3dVector(source_points)
    target.points = o3d.utility.Vector3dVector(target_points)

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    return result.transformation