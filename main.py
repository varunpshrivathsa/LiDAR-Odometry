import os
import argparse

from src.odometry import run_lidar_odometry_live


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence",
        type=str,
        default="07",
        help="KITTI sequence id, e.g. 00, 01, ..., 10"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/data/datasets/kitti/sequences",
        help="Root folder containing KITTI sequence folders with velodyne/"
    )
    parser.add_argument(
        "--calib_root",
        type=str,
        default="/data/datasets/kitti/calibration/sequences",
        help="Root folder containing KITTI calibration folders with calib.txt"
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        default="/data/datasets/kitti/groundtruth/poses",
        help="Root folder containing KITTI ground truth pose txt files"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame step size"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.001,
        help="Pause delay per frame for live display"
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.8,
        help="Voxel size for downsampling"
    )
    parser.add_argument(
        "--min_range",
        type=float,
        default=2.0,
        help="Minimum LiDAR range to keep"
    )
    parser.add_argument(
        "--max_range",
        type=float,
        default=60.0,
        help="Maximum LiDAR range to keep"
    )
    parser.add_argument(
        "--z_min",
        type=float,
        default=-2.5,
        help="Minimum z to keep in Velodyne frame"
    )
    parser.add_argument(
        "--z_max",
        type=float,
        default=1.5,
        help="Maximum z to keep in Velodyne frame"
    )
    parser.add_argument(
        "--scan_x_min",
        type=float,
        default=-10.0,
        help="Left plot x min (forward axis)"
    )
    parser.add_argument(
        "--scan_x_max",
        type=float,
        default=60.0,
        help="Left plot x max (forward axis)"
    )
    parser.add_argument(
        "--scan_y_min",
        type=float,
        default=-30.0,
        help="Left plot y min (left-right axis)"
    )
    parser.add_argument(
        "--scan_y_max",
        type=float,
        default=30.0,
        help="Left plot y max (left-right axis)"
    )

    args = parser.parse_args()

    sequence_path = os.path.join(args.dataset_root, args.sequence)
    calib_path = os.path.join(args.calib_root, args.sequence, "calib.txt")
    gt_path = os.path.join(args.gt_root, f"{args.sequence}.txt")

    run_lidar_odometry_live(
        sequence_path=sequence_path,
        calib_path=calib_path,
        gt_path=gt_path,
        sequence_name=args.sequence,
        max_frames=args.max_frames,
        step=args.step,
        playback_delay=args.delay,
        voxel_size=args.voxel_size,
        min_range=args.min_range,
        max_range=args.max_range,
        z_min=args.z_min,
        z_max=args.z_max,
        scan_x_min=args.scan_x_min,
        scan_x_max=args.scan_x_max,
        scan_y_min=args.scan_y_min,
        scan_y_max=args.scan_y_max
    )


if __name__ == "__main__":
    main()