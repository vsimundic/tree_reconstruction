#!/usr/bin/env python3
"""Export individual presentation images from one reconstruction capture.

The marker stages are recomputed for one frame with the production detector.
Camera diagrams use the saved pose sidecar. Existing non-rigid and evaluation
figures are copied without placing them inside slide-sized canvases.

Edit the variables below, then run this file without arguments.
"""

from pathlib import Path
import re
import shutil

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import depth_regions as dr
import marker_detection_sam3 as marker_detector


# ---------------------------------------------------------------------------
# Configuration -- edit these variables, then run this file without arguments.
# ---------------------------------------------------------------------------
CAPTURE_ROOT = Path("data/pears/pears_filtered_clean/B/tree_0_V_0000")
RECONSTRUCTION_DIR = Path("data/pears/pears_reconstructed/B/tree_0_V_0000")
CAPTURE_NAME = "tree_0_V_0000"
MARKER_FRAME_ID = "0"
OUTPUT_DIR = RECONSTRUCTION_DIR / "presentation_subimages"
PAIR_NEIGHBOR_SPAN = 3

BACKGROUND = "#0d1420"
PANEL = "#182334"
CYAN = "#3dc9ec"
GREEN = "#54d38c"
ORANGE = "#ffa64c"


def _slug(title: str) -> str:
    title = re.sub(r"^\d+\.\s*", "", title)
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def export_marker_steps() -> list[Path]:
    """Run the production marker pipeline and save each debug panel alone."""
    frame_dir = CAPTURE_ROOT / "marker_captures" / "depth"
    intensity, intensity8, depth_m, K = dr.load_frame(str(frame_dir), MARKER_FRAME_ID)
    panels, _, results = marker_detector.run_pipeline(depth_m, intensity, intensity8, K)
    if not results:
        raise RuntimeError(f"no 3D marker fit was accepted for frame {MARKER_FRAME_ID}")

    marker_dir = OUTPUT_DIR / "slide_02_marker"
    marker_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, (title, image) in enumerate(panels, start=1):
        path = marker_dir / f"{index:02d}_{_slug(title)}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"failed to save {path}")
        outputs.append(path)
    return outputs


def _load_poses() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    path = RECONSTRUCTION_DIR / f"{CAPTURE_NAME}_merged.poses.npz"
    with np.load(path, allow_pickle=False) as poses:
        frame_ids = [str(value) for value in poses["frame_ids"]]
        final = np.asarray(poses["final_poses"], dtype=np.float64)
        marker = np.asarray(poses["T_marker_cam"], dtype=np.float64)
    marker_ref = marker[0]
    initial = np.stack([marker_ref @ np.linalg.inv(value) for value in marker])
    return frame_ids, initial, final, marker_ref[:3, 3]


def _equal_3d_limits(axis, groups: list[np.ndarray]) -> None:
    points = np.concatenate(groups, axis=0)
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    radius = max(np.ptp(points, axis=0).max() / 2, 0.1)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def export_camera_pose_images(frame_ids: list[str], initial: np.ndarray,
                              final: np.ndarray, marker_center: np.ndarray) -> list[Path]:
    output = OUTPUT_DIR / "slide_03_camera_poses"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    plt.style.use("dark_background")
    all_centers = [initial[:, :3, 3], final[:, :3, 3], marker_center[None, :]]
    for filename, title, poses, colour in (
            ("01_marker_initial_camera_poses.png", "Marker-based initial camera poses",
             initial, CYAN),
            ("02_pose_graph_optimized_camera_poses.png", "Pose-graph optimized camera poses",
             final, GREEN)):
        fig = plt.figure(figsize=(10, 8), dpi=180, facecolor=BACKGROUND)
        axis = fig.add_subplot(111, projection="3d", facecolor=BACKGROUND)
        centers = poses[:, :3, 3]
        axis.plot(*centers.T, color=colour, linewidth=2, alpha=0.8)
        axis.scatter(*centers.T, color=colour, s=60, edgecolor="white", linewidth=0.7)
        for frame_id, center, pose in zip(frame_ids, centers, poses):
            axis.text(*center, frame_id, color="white", fontsize=9)
            forward = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
            axis.quiver(*center, *(0.12 * forward), color=colour, linewidth=1.2)
        axis.scatter(*marker_center, color=ORANGE, marker="s", s=110,
                     edgecolor="white", linewidth=0.8, label="shared marker")
        axis.set_title(title, fontsize=18, pad=18)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.legend(loc="upper left")
        axis.grid(alpha=0.18)
        _equal_3d_limits(axis, all_centers)
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        results.append(path)
    return results


def export_pose_graph(frame_ids: list[str], final: np.ndarray) -> Path:
    """Draw the actual camera layout and the neighbor pairs tested by ICP."""
    output = OUTPUT_DIR / "slide_04_icp_pose_graph"
    output.mkdir(parents=True, exist_ok=True)
    centers = final[:, :3, 3]
    centered = centers - centers.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    points = centered @ vh[:2].T

    plt.style.use("dark_background")
    fig, axis = plt.subplots(figsize=(11, 8), dpi=180, facecolor=BACKGROUND)
    axis.set_facecolor(PANEL)
    for i in range(len(frame_ids)):
        for j in range(i + 1, min(len(frame_ids), i + PAIR_NEIGHBOR_SPAN + 1)):
            consecutive = j == i + 1
            axis.plot(points[[i, j], 0], points[[i, j], 1],
                      color=CYAN if consecutive else ORANGE,
                      linewidth=3 if consecutive else 1.8,
                      linestyle="-" if consecutive else "--", alpha=0.85)
    axis.scatter(points[:, 0], points[:, 1], s=230, color=GREEN,
                 edgecolor="white", linewidth=1.3, zorder=4)
    for frame_id, point in zip(frame_ids, points):
        axis.text(*point, frame_id, ha="center", va="center", color=BACKGROUND,
                  fontsize=11, fontweight="bold", zorder=5)
    axis.plot([], [], color=CYAN, linewidth=3, label="consecutive pair")
    axis.plot([], [], color=ORANGE, linewidth=2, linestyle="--",
              label="additional neighbor pair")
    axis.set_title("Camera pairs tested by point-to-plane ICP", fontsize=18, pad=16)
    axis.set_xlabel("camera-layout axis 1")
    axis.set_ylabel("camera-layout axis 2")
    axis.legend(loc="best")
    axis.grid(alpha=0.15)
    axis.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    path = output / "01_icp_neighbor_pairs_and_pose_graph.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def export_existing_algorithm_outputs() -> list[Path]:
    """Copy raw code outputs with slide-oriented names; do not compose them."""
    nonrigid = RECONSTRUCTION_DIR / "non_rigid_comparison"
    projection = (RECONSTRUCTION_DIR / "projection_fit_comparison" /
                  f"{CAPTURE_NAME}_merged_scene")
    sources = {
        "slide_05_rigid_result/01_rigid_vs_nonrigid_front.png":
            nonrigid / f"{CAPTURE_NAME}_merged_scene_comparison_front.png",
        "slide_05_rigid_result/02_rigid_vs_nonrigid_side.png":
            nonrigid / f"{CAPTURE_NAME}_merged_scene_comparison_side.png",
        "slide_06_mobility/01_branch_tip_mobility.png":
            nonrigid / f"{CAPTURE_NAME}_merged_scene_mobility_front.png",
        "slide_07_nonrigid/01_branch_tip_movement.png":
            nonrigid / f"{CAPTURE_NAME}_merged_scene_movement_front.png",
        "slide_08_evaluation/01_rigid_vs_nonrigid_all_views.png":
            projection / "rigid_vs_nonrigid_all_frames.png",
        "slide_08_evaluation/02_projection_comparison_frame_0.png":
            projection / "comparison" / "frame_0.png",
        "slide_08_evaluation/03_metrics_comparison.png":
            projection / "metrics_comparison.png",
    }
    outputs = []
    for destination, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        path = OUTPUT_DIR / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, path)
        outputs.append(path)
    return outputs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = export_marker_steps()
    frame_ids, initial, final, marker_center = _load_poses()
    outputs.extend(export_camera_pose_images(
        frame_ids, initial, final, marker_center))
    outputs.append(export_pose_graph(frame_ids, final))
    outputs.extend(export_existing_algorithm_outputs())
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
