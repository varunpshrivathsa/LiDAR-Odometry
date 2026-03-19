from src.kitti_loader import KITTILoader
from src.preprocess import voxel_downsample
from src.registration import run_icp
from src.trajectory import accumulate_trajectory
from src.plotting import plot_trajectory_2d


def main():
    sequence_path = "/data/datasets/kitti/sequences/07"
    loader = KITTILoader(sequence_path)

    relative_transforms = []

    for i in range(len(loader) - 1):
        scan1 = voxel_downsample(loader.get_point_cloud(i), voxel_size=0.2)
        scan2 = voxel_downsample(loader.get_point_cloud(i + 1), voxel_size=0.2)

        T = run_icp(scan1, scan2)
        relative_transforms.append(T)

        print(f"Processed frame {i} -> {i+1}")

    poses = accumulate_trajectory(relative_transforms)
    plot_trajectory_2d(poses)
    print("Total poses:", len(poses))


if __name__ == "__main__":
    main()