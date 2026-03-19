import os
import glob
import numpy as np


def load_velodyne_paths(sequence_path):
    velo_dir = os.path.join(sequence_path, "velodyne")
    velo_files = sorted(glob.glob(os.path.join(velo_dir, "*.bin")))
    if len(velo_files) == 0:
        raise FileNotFoundError(f"No .bin files found in {velo_dir}")
    return velo_files


def load_ground_truth(gt_path):
    poses = []
    with open(gt_path, "r") as f:
        for line in f:
            vals = np.fromstring(line.strip(), sep=" ")
            if vals.size != 12:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = vals.reshape(3, 4)
            poses.append(T)
    return poses


def read_calib_kitti(calib_path):
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.strip().split(":", 1)
            data[key] = np.array([float(x) for x in val.strip().split()], dtype=np.float64)

    if "Tr" in data:
        Tr = data["Tr"].reshape(3, 4)
    elif "Tr_velo_to_cam" in data:
        Tr = data["Tr_velo_to_cam"].reshape(3, 4)
    else:
        raise KeyError(f"Could not find 'Tr' or 'Tr_velo_to_cam' in {calib_path}")

    T_cam_velo = np.eye(4, dtype=np.float64)
    T_cam_velo[:3, :4] = Tr
    return T_cam_velo


def load_kitti_scan(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]