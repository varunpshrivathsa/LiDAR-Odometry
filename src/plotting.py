import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_2d(poses):
    traj = np.array([[pose[0, 3], pose[1, 3]] for pose in poses])
    plt.figure(figsize=(8, 6))
    plt.plot(traj[:, 0], traj[:, 1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Estimated LiDAR Odometry Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.show()