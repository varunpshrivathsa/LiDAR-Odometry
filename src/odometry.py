import numpy as np
import matplotlib.pyplot as plt

from src.data_utils import load_velodyne_paths, load_ground_truth, read_calib_kitti, load_kitti_scan
from src.processing import preprocess_points, icp_motion
from src.metrics import compute_all_metrics, format_metrics_text
from src.visualization import (
    setup_live_figure,
    update_scan_plot,
    update_trajectory_plot,
    show_final_metrics_in_plot,
)


def run_lidar_odometry_live(
    sequence_path,
    calib_path,
    gt_path,
    sequence_name="07",
    max_frames=None,
    step=1,
    playback_delay=0.001,
    voxel_size=0.8,
    min_range=2.0,
    max_range=60.0,
    z_min=-2.5,
    z_max=1.5,
    scan_x_min=-10.0,
    scan_x_max=60.0,
    scan_y_min=-30.0,
    scan_y_max=30.0
):
    velo_files = load_velodyne_paths(sequence_path)
    gt_poses = load_ground_truth(gt_path)
    T_cam_velo = read_calib_kitti(calib_path)
    T_velo_cam = np.linalg.inv(T_cam_velo)

    n = min(len(velo_files), len(gt_poses))
    if max_frames is not None:
        n = min(n, max_frames)

    pose_cam = np.eye(4, dtype=np.float64)
    est_traj = [pose_cam[:3, 3].copy()]
    gt_all = np.asarray([p[:3, 3].copy() for p in gt_poses[:n]])

    (
        fig,
        ax_scan,
        ax_traj,
        scan_artist,
        car_pt,
        forward_line,
        gt_line,
        est_line,
        current_gt_pt,
        current_est_pt,
        metrics_text_artist,
        scan_view,
    ) = setup_live_figure(
        gt_all,
        sequence_name,
        scan_x_min,
        scan_x_max,
        scan_y_min,
        scan_y_max,
    )

    print(f"Total frames used: {n}")
    print(f"Calibration file: {calib_path}")
    print("Controls:")
    print("  Left mouse drag on LiDAR plot  -> pan")
    print("  Mouse wheel on LiDAR plot      -> zoom")
    print("  Press r                        -> reset LiDAR view")
    print("  Ctrl+C in terminal             -> stop")

    try:
        for i in range(0, n - step, step):
            src_raw = load_kitti_scan(velo_files[i])
            tgt_raw = load_kitti_scan(velo_files[i + step])

            src = preprocess_points(
                src_raw,
                min_range=min_range,
                max_range=max_range,
                z_min=z_min,
                z_max=z_max,
                voxel_size=voxel_size
            )
            tgt = preprocess_points(
                tgt_raw,
                min_range=min_range,
                max_range=max_range,
                z_min=z_min,
                z_max=z_max,
                voxel_size=voxel_size
            )

            update_scan_plot(scan_artist, tgt)

            if len(src) < 100 or len(tgt) < 100:
                est_traj.append(pose_cam[:3, 3].copy())
                est_arr = np.asarray(est_traj)
                update_trajectory_plot(
                    ax_traj=ax_traj,
                    gt_all=gt_all,
                    est_arr=est_arr,
                    est_line=est_line,
                    current_gt_pt=current_gt_pt,
                    current_est_pt=current_est_pt,
                    frame_idx=min(i + step, n - 1),
                )
                fig.canvas.draw_idle()
                plt.pause(playback_delay)
                print(f"[Frame {i:04d}] skipped: too few points after preprocessing")
                continue

            T_velo_t_to_t1, fitness, rmse = icp_motion(src, tgt, voxel_size=voxel_size)
            T_cam_t_to_t1 = T_cam_velo @ T_velo_t_to_t1 @ T_velo_cam
            pose_cam = pose_cam @ np.linalg.inv(T_cam_t_to_t1)
            est_traj.append(pose_cam[:3, 3].copy())

            trans_step = np.linalg.norm(T_cam_t_to_t1[:3, 3])

            print(
                f"[Frame {i:04d}] "
                f"src={len(src):5d}, "
                f"tgt={len(tgt):5d}, "
                f"fitness={fitness:.4f}, "
                f"rmse={rmse:.4f}, "
                f"|t|={trans_step:.4f}"
            )

            est_arr = np.asarray(est_traj)
            update_trajectory_plot(
                ax_traj=ax_traj,
                gt_all=gt_all,
                est_arr=est_arr,
                est_line=est_line,
                current_gt_pt=current_gt_pt,
                current_est_pt=current_est_pt,
                frame_idx=i + step,
            )

            scan_view.apply_limits()
            fig.canvas.draw_idle()
            plt.pause(playback_delay)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    est_arr = np.asarray(est_traj)
    metrics = compute_all_metrics(est_arr, gt_all)
    metrics_str = format_metrics_text(metrics)

    print("\nFinal Metrics")
    print("------------------------------")
    print(metrics_str)

    show_final_metrics_in_plot(metrics_text_artist, metrics_str)
    fig.canvas.draw_idle()

    plt.ioff()
    plt.show()