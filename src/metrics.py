import numpy as np


def compute_ate_rmse(est_xyz, gt_xyz):
    n = min(len(est_xyz), len(gt_xyz))
    if n == 0:
        return np.nan
    err = est_xyz[:n] - gt_xyz[:n]
    se = np.sum(err ** 2, axis=1)
    return float(np.sqrt(np.mean(se)))


def compute_final_translation_error(est_xyz, gt_xyz):
    n = min(len(est_xyz), len(gt_xyz))
    if n == 0:
        return np.nan
    return float(np.linalg.norm(est_xyz[n - 1] - gt_xyz[n - 1]))


def compute_mean_relative_translation_error(est_xyz, gt_xyz):
    n = min(len(est_xyz), len(gt_xyz))
    if n < 2:
        return np.nan

    est_rel = est_xyz[1:n] - est_xyz[:n - 1]
    gt_rel = gt_xyz[1:n] - gt_xyz[:n - 1]

    rel_err = np.linalg.norm(est_rel - gt_rel, axis=1)
    return float(np.mean(rel_err))


def compute_all_metrics(est_xyz, gt_xyz):
    return {
        "ATE RMSE": compute_ate_rmse(est_xyz, gt_xyz),
        "Final Translation Error": compute_final_translation_error(est_xyz, gt_xyz),
        "Mean Relative Translation Error": compute_mean_relative_translation_error(est_xyz, gt_xyz),
    }


def format_metrics_text(metrics):
    return (
        f"ATE RMSE: {metrics['ATE RMSE']:.4f} m\n"
        f"Final Translation Error: {metrics['Final Translation Error']:.4f} m\n"
        f"Mean Relative Translation Error: {metrics['Mean Relative Translation Error']:.4f} m"
    )