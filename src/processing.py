import numpy as np
import open3d as o3d


def preprocess_points(
    points,
    min_range=2.0,
    max_range=60.0,
    z_min=-2.5,
    z_max=1.5,
    voxel_size=0.8
):
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    ranges = np.linalg.norm(points, axis=1)
    mask = (
        (ranges >= min_range) &
        (ranges <= max_range) &
        (points[:, 2] >= z_min) &
        (points[:, 2] <= z_max)
    )
    pts = points[mask]

    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    if len(pcd.points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    return np.asarray(pcd.points)


def make_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def estimate_normals(pcd, radius):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    return pcd


def icp_motion(src_points, tgt_points, voxel_size=0.8):
    src = make_pcd(src_points)
    tgt = make_pcd(tgt_points)

    estimate_normals(src, radius=voxel_size * 2.0)
    estimate_normals(tgt, radius=voxel_size * 2.0)

    init = np.eye(4, dtype=np.float64)

    reg_coarse = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_correspondence_distance=voxel_size * 3.0,
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    reg_fine = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_correspondence_distance=voxel_size * 1.5,
        init=reg_coarse.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    return reg_fine.transformation, reg_fine.fitness, reg_fine.inlier_rmse