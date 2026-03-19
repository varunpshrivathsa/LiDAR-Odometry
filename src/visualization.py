import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")


class InteractiveScanView:
    def __init__(self, ax, xlim=(-10.0, 60.0), ylim=(-30.0, 30.0)):
        self.ax = ax
        self.default_xlim = list(xlim)
        self.default_ylim = list(ylim)

        self.xlim = list(xlim)
        self.ylim = list(ylim)

        self.dragging = False
        self.last_event = None

        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)

        self.cid_press = ax.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_scroll = ax.figure.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.cid_key = ax.figure.canvas.mpl_connect("key_press_event", self.on_key)

    def apply_limits(self):
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self.dragging = True
            self.last_event = event

    def on_release(self, event):
        if event.button == 1:
            self.dragging = False
            self.last_event = None

    def on_motion(self, event):
        if not self.dragging or self.last_event is None:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.last_event.xdata is None or self.last_event.ydata is None:
            self.last_event = event
            return

        dx = event.xdata - self.last_event.xdata
        dy = event.ydata - self.last_event.ydata

        self.xlim[0] -= dx
        self.xlim[1] -= dx
        self.ylim[0] -= dy
        self.ylim[1] -= dy

        self.apply_limits()
        self.ax.figure.canvas.draw_idle()
        self.last_event = event

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        scale = 0.9 if event.button == "up" else 1.1

        x0, x1 = self.xlim
        y0, y1 = self.ylim

        width = (x1 - x0) * scale
        height = (y1 - y0) * scale

        cx = event.xdata
        cy = event.ydata

        relx = (cx - x0) / max(1e-9, (x1 - x0))
        rely = (cy - y0) / max(1e-9, (y1 - y0))

        new_x0 = cx - relx * width
        new_x1 = new_x0 + width
        new_y0 = cy - rely * height
        new_y1 = new_y0 + height

        self.xlim = [new_x0, new_x1]
        self.ylim = [new_y0, new_y1]

        self.apply_limits()
        self.ax.figure.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "r":
            self.xlim = self.default_xlim.copy()
            self.ylim = self.default_ylim.copy()
            self.apply_limits()
            self.ax.figure.canvas.draw_idle()


def setup_live_figure(gt_all, sequence_name, scan_x_min, scan_x_max, scan_y_min, scan_y_max):
    plt.ion()
    fig, (ax_scan, ax_traj) = plt.subplots(1, 2, figsize=(16, 7))

    fig.patch.set_facecolor("black")
    ax_scan.set_facecolor("black")
    ax_traj.set_facecolor("black")

    scan_artist = ax_scan.scatter([], [], s=0.5, c="white")
    car_pt = ax_scan.scatter([0], [0], s=60, marker="o", c="red")
    forward_line, = ax_scan.plot([0, 4], [0, 0], linewidth=2, color="yellow")

    ax_scan.set_title("Current LiDAR Scan (Vehicle-Centered Top View)", color="white")
    ax_scan.set_xlabel("Forward X (m)", color="white")
    ax_scan.set_ylabel("Left Y (m)", color="white")
    ax_scan.set_aspect("equal", adjustable="box")
    ax_scan.grid(True, color="gray", linestyle="--", linewidth=0.5)
    ax_scan.set_xlim(scan_x_min, scan_x_max)
    ax_scan.set_ylim(scan_y_min, scan_y_max)

    gt_line, = ax_traj.plot(
        gt_all[:, 0], gt_all[:, 2],
        color="blue",
        label="Ground Truth",
        linewidth=2
    )
    est_line, = ax_traj.plot(
        [], [],
        color="red",
        label="Estimated",
        linewidth=2
    )
    current_gt_pt, = ax_traj.plot(
        [gt_all[0, 0]], [gt_all[0, 2]],
        "o",
        color="blue",
        markersize=5
    )
    current_est_pt, = ax_traj.plot(
        [], [],
        "o",
        color="red",
        markersize=5
    )

    metrics_text = ax_traj.text(
        0.02,
        0.98,
        "",
        transform=ax_traj.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", edgecolor="white", alpha=0.8)
    )

    ax_traj.set_title(f"LiDAR Odometry Live Trajectory - KITTI {sequence_name}", color="white")
    ax_traj.set_xlabel("X", color="white")
    ax_traj.set_ylabel("Z", color="white")
    ax_traj.grid(True, color="gray", linestyle="--", linewidth=0.5)
    ax_traj.axis("equal")

    for ax in [ax_scan, ax_traj]:
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

    legend = ax_traj.legend()
    for text in legend.get_texts():
        text.set_color("white")
    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")

    scan_view = InteractiveScanView(
        ax_scan,
        xlim=(scan_x_min, scan_x_max),
        ylim=(scan_y_min, scan_y_max),
    )

    plt.tight_layout()

    return (
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
        metrics_text,
        scan_view,
    )


def update_scan_plot(scan_artist, points_local):
    if points_local.shape[0] == 0:
        scan_artist.set_offsets(np.empty((0, 2)))
        return

    xy = points_local[:, :2]
    scan_artist.set_offsets(xy)


def update_trajectory_plot(ax_traj, gt_all, est_arr, est_line, current_gt_pt, current_est_pt, frame_idx):
    est_line.set_data(est_arr[:, 0], est_arr[:, 2])
    current_gt_pt.set_data([gt_all[frame_idx, 0]], [gt_all[frame_idx, 2]])
    current_est_pt.set_data([est_arr[-1, 0]], [est_arr[-1, 2]])

    x_min, x_max = gt_all[:, 0].min(), gt_all[:, 0].max()
    z_min_plot, z_max_plot = gt_all[:, 2].min(), gt_all[:, 2].max()
    margin_x = 0.1 * max(1e-6, (x_max - x_min))
    margin_z = 0.1 * max(1e-6, (z_max_plot - z_min_plot))

    ax_traj.set_xlim(x_min - margin_x, x_max + margin_x)
    ax_traj.set_ylim(z_min_plot - margin_z, z_max_plot + margin_z)


def show_final_metrics_in_plot(metrics_text_artist, metrics_text):
    metrics_text_artist.set_text(metrics_text)