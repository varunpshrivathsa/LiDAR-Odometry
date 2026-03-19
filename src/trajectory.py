import numpy as np


def accumulate_trajectory(relative_transforms):
    poses = [np.eye(4)]
    current_pose = np.eye(4)

    for T in relative_transforms:
        current_pose = current_pose @ T
        poses.append(current_pose.copy())

    return poses