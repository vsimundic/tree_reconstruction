#!/usr/bin/env python3
"""Standalone, conservative non-rigid refinement for a reconstructed tree.

This experiment consumes an existing ``*_merged_scene.ply`` and its
``*_merged.poses.npz`` sidecar.  It does not import or alter the reconstruction
pipeline.  The uniform per-frame colours written by the pipeline recover the
contributing point clouds.

Only likely branch ends may move.  A node needs low cross-frame support, a thin
line-like neighborhood, and low local density before it gets any mobility.
Stable nodes in every frame anchor trunks, poles, markers, and the lower scene.
Eligible tips move smoothly toward a frame-balanced median consensus, with a
hard displacement cap.

Outputs include a deformed PLY, displacement data, alignment metrics, and
headless before/after PNG comparisons.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix, csr_matrix, diags, vstack
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Configuration -- edit these variables, then run this file without arguments.
# ---------------------------------------------------------------------------
INPUT_PLY = Path(
    "data/pears/pears_reconstructed/B/tree_0_V_0000/tree_0_V_0000_merged_scene.ply")
POSES_PATH: Path | None = None  # None finds the matching *_merged.poses.npz.
OUTPUT_DIR: Path | None = None  # None creates non_rigid_comparison/ beside the PLY.

WORK_VOXEL_M = 0.018
NODE_SPACING_M = 0.08
BLEND_NODES = 4
GRAPH_NEIGHBORS = 6

# A node must disagree across views and look like a thin, sparse branch end.
STATIC_SUPPORT_RADIUS_M = 0.012
STATIC_SUPPORT_FRAMES = 3
SHAPE_RADIUS_M = 0.075
LINEARITY_START = 0.50
LINEARITY_FULL = 0.80
DENSITY_FULL_BELOW = 35
DENSITY_ZERO_ABOVE = 75

# Start wide enough to find moved tips, then tighten the matches.
CORRESPONDENCE_RADII_M = (0.12, 0.12, 0.08, 0.08, 0.05, 0.05, 0.035, 0.035)
MIN_CONSENSUS_FRAMES = 2
MAX_CONSENSUS_SPREAD_FRACTION = 0.70
MIN_POINT_MOBILITY = 0.03
DATA_WEIGHT = 4.0
SMOOTHNESS = 0.15
TIP_MAGNITUDE = 0.005
STATIC_ANCHOR = 120.0
BOTTOM_ANCHOR_FRACTION = 0.20
MAX_TIP_DISPLACEMENT_M = 0.08
MAX_RENDER_POINTS = 180_000


@dataclass(frozen=True)
class Settings:
    work_voxel_m: float
    node_spacing_m: float
    blend_nodes: int
    graph_neighbors: int
    static_support_radius_m: float
    static_support_frames: int
    shape_radius_m: float
    linearity_start: float
    linearity_full: float
    density_full_below: int
    density_zero_above: int
    correspondence_radii_m: tuple[float, ...]
    min_consensus_frames: int
    max_consensus_spread_fraction: float
    min_point_mobility: float
    data_weight: float
    smoothness: float
    tip_magnitude: float
    static_anchor: float
    bottom_anchor_fraction: float
    max_tip_displacement_m: float


@dataclass
class FrameCloud:
    frame_id: str
    rgb: np.ndarray
    original: np.ndarray
    work_original: np.ndarray
    nodes: np.ndarray
    blend_full: csr_matrix
    blend_work: csr_matrix
    smooth_matrix: csr_matrix
    node_height: np.ndarray
    node_support: np.ndarray
    node_linearity: np.ndarray
    node_density: np.ndarray
    node_mobility: np.ndarray
    node_displacement: np.ndarray

    def full_deformed(self) -> np.ndarray:
        return self.original + self.blend_full @ self.node_displacement

    def work_deformed(self) -> np.ndarray:
        return self.work_original + self.blend_work @ self.node_displacement


def _normalise(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + 1e-12)


def _default_poses_path(input_path: Path) -> Path:
    """Return *_merged.poses.npz for merged or merged-scene input."""
    stem = input_path.stem
    if stem.endswith("_merged_scene"):
        stem = stem.removesuffix("_scene")
    return input_path.with_name(f"{stem}.poses.npz")


def _load_up_and_frame_ids(poses_path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    if not poses_path.is_file():
        raise FileNotFoundError(f"pose sidecar not found: {poses_path}")
    with np.load(poses_path, allow_pickle=False) as poses:
        required = {"frame_ids", "final_poses"}
        missing = required.difference(poses.files)
        if missing:
            raise ValueError(f"pose sidecar is missing: {sorted(missing)}")
        frame_ids = [str(value) for value in poses["frame_ids"]]
        final_poses = np.asarray(poses["final_poses"], dtype=np.float64)
    if final_poses.shape != (len(frame_ids), 4, 4):
        raise ValueError("final_poses and frame_ids have incompatible shapes")
    camera_down = final_poses[:, :3, :3] @ np.array([0.0, 1.0, 0.0])
    up = _normalise(-camera_down.mean(axis=0))
    return up, frame_ids, final_poses


def _split_frame_colours(
        points: np.ndarray, colours: np.ndarray, frame_ids: list[str]
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Recover frame clouds from contiguous uniform-colour blocks in the PLY."""
    rgb = np.rint(np.clip(colours, 0.0, 1.0) * 255.0).astype(np.uint8)
    if len(rgb) == 0:
        raise ValueError("input PLY has no points")
    starts = np.flatnonzero(np.r_[True, np.any(rgb[1:] != rgb[:-1], axis=1)])
    ends = np.r_[starts[1:], len(rgb)]
    if len(starts) != len(frame_ids):
        unique = np.unique(rgb, axis=0)
        raise ValueError(
            "could not recover one contiguous colour block per frame: "
            f"found {len(starts)} blocks and {len(unique)} colours for "
            f"{len(frame_ids)} pose entries")
    result = []
    for frame_id, start, end in zip(frame_ids, starts, ends):
        result.append((frame_id, rgb[start].copy(), points[start:end].copy()))
    return result


def _voxel_representatives(points: np.ndarray, voxel_m: float) -> np.ndarray:
    """Pick one original point per occupied voxel in deterministic order."""
    origin = points.min(axis=0)
    cells = np.floor((points - origin) / voxel_m).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)]


def _blend_matrix(points: np.ndarray, nodes: np.ndarray, count: int,
                  spacing_m: float) -> csr_matrix:
    count = min(max(1, count), len(nodes))
    distances, indices = cKDTree(nodes).query(points, k=count)
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = np.exp(-0.5 * (distances / spacing_m) ** 2)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-12
    rows = np.repeat(np.arange(len(points)), count)
    return coo_matrix(
        (weights.ravel(), (rows, indices.ravel())),
        shape=(len(points), len(nodes))).tocsr()


def _smoothness_matrix(nodes: np.ndarray, neighbor_count: int) -> csr_matrix:
    if len(nodes) < 2:
        return csr_matrix((0, len(nodes)))
    count = min(max(1, neighbor_count), len(nodes) - 1)
    _, near = cKDTree(nodes).query(nodes, k=count + 1)
    edges = set()
    for source, neighbors in enumerate(near[:, 1:]):
        for target in np.atleast_1d(neighbors):
            edge = tuple(sorted((source, int(target))))
            if edge[0] != edge[1]:
                edges.add(edge)
    rows = []
    cols = []
    values = []
    for row, (first, second) in enumerate(sorted(edges)):
        rows.extend((row, row))
        cols.extend((first, second))
        values.extend((1.0, -1.0))
    return coo_matrix(
        (values, (rows, cols)), shape=(len(edges), len(nodes))).tocsr()


def _prepare_frames(
        blocks: list[tuple[str, np.ndarray, np.ndarray]], up: np.ndarray,
        settings: Settings) -> list[FrameCloud]:
    all_points = np.concatenate([block[2] for block in blocks], axis=0)
    low = float(np.percentile(all_points @ up, 1.0))
    high = float(np.percentile(all_points @ up, 99.0))
    height_range = max(high - low, 1e-6)
    frames = []
    for frame_id, rgb, original in blocks:
        work = _voxel_representatives(original, settings.work_voxel_m)
        nodes = _voxel_representatives(work, settings.node_spacing_m)
        node_height = np.clip((nodes @ up - low) / height_range, 0.0, 1.0)
        count = len(nodes)
        frames.append(FrameCloud(
            frame_id=frame_id,
            rgb=rgb,
            original=original,
            work_original=work,
            nodes=nodes,
            blend_full=_blend_matrix(
                original, nodes, settings.blend_nodes, settings.node_spacing_m),
            blend_work=_blend_matrix(
                work, nodes, settings.blend_nodes, settings.node_spacing_m),
            smooth_matrix=_smoothness_matrix(nodes, settings.graph_neighbors),
            node_height=node_height,
            node_support=np.zeros(count, dtype=np.int32),
            node_linearity=np.zeros(count, dtype=np.float64),
            node_density=np.zeros(count, dtype=np.int32),
            node_mobility=np.zeros(count, dtype=np.float64),
            node_displacement=np.zeros((count, 3), dtype=np.float64)))
    _assign_branch_tip_mobility(frames, settings)
    return frames


def _local_shape_scores(frame: FrameCloud, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    neighborhoods = cKDTree(frame.work_original).query_ball_point(frame.nodes, radius_m)
    density = np.asarray([len(indices) for indices in neighborhoods], dtype=np.int32)
    linearity = np.zeros(len(frame.nodes), dtype=np.float64)
    for index, neighbors in enumerate(neighborhoods):
        if len(neighbors) < 8:
            continue
        local = frame.work_original[neighbors]
        covariance = np.cov(local - local.mean(axis=0), rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
        linearity[index] = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + 1e-12)
    return linearity, density


def _assign_branch_tip_mobility(frames: list[FrameCloud], settings: Settings) -> None:
    """Anchor stable/thick geometry and give only unsupported thin ends mobility."""
    work_trees = [cKDTree(frame.work_original) for frame in frames]
    linearity_width = max(settings.linearity_full - settings.linearity_start, 1e-6)
    density_width = max(settings.density_zero_above - settings.density_full_below, 1)
    for index, frame in enumerate(frames):
        support = np.zeros(len(frame.nodes), dtype=np.int32)
        for other, tree in enumerate(work_trees):
            if other == index:
                continue
            distance, _ = tree.query(frame.nodes, k=1)
            support += distance < settings.static_support_radius_m
        linearity, density = _local_shape_scores(frame, settings.shape_radius_m)

        support_gate = np.clip(
            (settings.static_support_frames - support) / settings.static_support_frames,
            0.0, 1.0)
        linearity_gate = np.clip(
            (linearity - settings.linearity_start) / linearity_width, 0.0, 1.0)
        density_gate = np.clip(
            (settings.density_zero_above - density) / density_width, 0.0, 1.0)
        mobility = support_gate * linearity_gate * density_gate
        mobility[frame.node_height <= settings.bottom_anchor_fraction] = 0.0

        frame.node_support = support
        frame.node_linearity = linearity
        frame.node_density = density
        frame.node_mobility = mobility


def _consensus_correspondences(
        source: np.ndarray, target_groups: list[np.ndarray], radius_m: float,
        point_mobility: np.ndarray, settings: Settings
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use one nearest candidate per frame, then take their median position."""
    distances = []
    candidates = []
    for target in target_groups:
        distance, target_indices = cKDTree(target).query(source, k=1)
        distances.append(distance)
        candidates.append(target[target_indices])
    distances_array = np.stack(distances, axis=1)
    candidates_array = np.stack(candidates, axis=1)
    inside = distances_array < radius_m

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        consensus = np.nanmedian(
            np.where(inside[:, :, None], candidates_array, np.nan), axis=1)
        spread = np.nanmedian(
            np.where(
                inside,
                np.linalg.norm(candidates_array - consensus[:, None, :], axis=2),
                np.nan),
            axis=1)

    support_count = inside.sum(axis=1)
    required = min(settings.min_consensus_frames, len(target_groups))
    valid = np.flatnonzero(
        (support_count >= required)
        & (point_mobility > settings.min_point_mobility)
        & np.isfinite(consensus[:, 0])
        & (spread < radius_m * settings.max_consensus_spread_fraction))
    if len(valid) == 0:
        return valid, consensus[valid], np.empty(0), np.empty(0)

    support_weight = np.minimum(support_count[valid], 5) / min(len(target_groups), 5)
    coherence_weight = np.exp(
        -0.5 * (spread[valid] / max(0.5 * radius_m, 1e-6)) ** 2)
    weights = settings.data_weight * (0.1 + 0.9 * point_mobility[valid])
    weights *= support_weight * coherence_weight
    residual = np.linalg.norm(consensus[valid] - source[valid], axis=1)
    return valid, consensus[valid], weights, residual


def _solve_frame(
        frame: FrameCloud, target_groups: list[np.ndarray], radius_m: float,
        settings: Settings) -> tuple[np.ndarray, dict[str, float]]:
    source_current = frame.work_deformed()
    point_mobility = np.asarray(frame.blend_work @ frame.node_mobility).ravel()
    source_indices, consensus, weights, residuals = _consensus_correspondences(
        source_current, target_groups, radius_m, point_mobility, settings)
    if len(source_indices) < max(30, len(frame.nodes) // 4):
        return frame.node_displacement.copy(), {
            "matches": float(len(source_indices)),
            "median_consensus_residual_m": (
                float(np.median(residuals)) if len(residuals) else float("nan"))}

    weighted_data = diags(np.sqrt(weights)) @ frame.blend_work[source_indices]
    target_displacement = consensus - frame.work_original[source_indices]
    weighted_target = np.sqrt(weights)[:, None] * target_displacement

    # Nodes with zero mobility are strong anchors. At a branch junction,
    # smoothness makes movement grow gradually toward the free end.
    magnitude_weights = settings.tip_magnitude
    magnitude_weights += settings.static_anchor * (1.0 - frame.node_mobility) ** 2
    regularise = diags(np.sqrt(magnitude_weights))
    smooth = np.sqrt(settings.smoothness) * frame.smooth_matrix
    system = vstack((weighted_data, smooth, regularise), format="csr")
    zeros = np.zeros((smooth.shape[0] + regularise.shape[0], 3))
    right_hand_side = np.vstack((weighted_target, zeros))

    displacement = np.empty((len(frame.nodes), 3), dtype=np.float64)
    for axis in range(3):
        displacement[:, axis] = lsqr(
            system, right_hand_side[:, axis], atol=1e-6, btol=1e-6,
            iter_lim=250)[0]

    lengths = np.linalg.norm(displacement, axis=1)
    limits = settings.max_tip_displacement_m * frame.node_mobility
    over_limit = lengths > limits
    displacement[over_limit] *= (limits[over_limit] / lengths[over_limit])[:, None]
    return displacement, {
        "matches": float(len(source_indices)),
        "median_consensus_residual_m": float(np.median(residuals)),
        "max_node_displacement_m": float(np.linalg.norm(displacement, axis=1).max())}


def optimise(frames: list[FrameCloud], settings: Settings) -> list[dict[str, object]]:
    """Coarse-to-fine, frame-balanced branch-tip deformation."""
    if len(frames) < 2:
        raise ValueError("non-rigid refinement needs at least two frame clouds")
    history: list[dict[str, object]] = []
    pass_count = len(settings.correspondence_radii_m)
    for pass_index, radius_m in enumerate(settings.correspondence_radii_m):
        snapshots = [frame.work_deformed() for frame in frames]
        updates = []
        pass_stats: dict[str, object] = {
            "pass": pass_index + 1, "correspondence_radius_m": radius_m, "frames": {}}
        for index, frame in enumerate(frames):
            targets = [cloud for other, cloud in enumerate(snapshots) if other != index]
            update, stats = _solve_frame(frame, targets, radius_m, settings)
            updates.append(update)
            pass_stats["frames"][frame.frame_id] = stats
        for frame, update in zip(frames, updates):
            frame.node_displacement = update
        history.append(pass_stats)
        residuals = [
            entry["median_consensus_residual_m"]
            for entry in pass_stats["frames"].values()]
        finite = [value for value in residuals if np.isfinite(value)]
        median_mm = 1000.0 * float(np.median(finite)) if finite else float("nan")
        print(f"pass {pass_index + 1}/{pass_count} (r={radius_m * 1000:.0f} mm): "
              f"median tip-to-consensus residual = {median_mm:.1f} mm")
    return history


def _alignment_metrics(
        point_groups: list[np.ndarray], threshold_m: float,
        masks: list[np.ndarray] | None = None) -> dict[str, float]:
    all_distances = []
    for index, source in enumerate(point_groups):
        selected = source if masks is None else source[masks[index]]
        if len(selected) == 0:
            continue
        target = np.concatenate(
            [points for other, points in enumerate(point_groups) if other != index], axis=0)
        distances, _ = cKDTree(target).query(selected, k=1)
        all_distances.append(distances)
    if not all_distances:
        return {
            "point_count": 0,
            "median_cross_frame_distance_m": float("nan"),
            "p90_cross_frame_distance_m": float("nan"),
            "fraction_with_neighbor_within_threshold": float("nan"),
            "threshold_m": threshold_m}
    distances = np.concatenate(all_distances)
    return {
        "point_count": len(distances),
        "median_cross_frame_distance_m": float(np.median(distances)),
        "p90_cross_frame_distance_m": float(np.percentile(distances, 90)),
        "fraction_with_neighbor_within_threshold": float(np.mean(distances <= threshold_m)),
        "threshold_m": threshold_m,
    }


def _view_basis(points: np.ndarray, up: np.ndarray,
                final_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre = np.median(points, axis=0)
    camera_centre = final_poses[:, :3, 3].mean(axis=0)
    front = centre - camera_centre
    front = front - up * np.dot(front, up)
    if np.linalg.norm(front) < 1e-6:
        centred = points - centre
        covariance = centred.T @ centred
        _, eigenvectors = np.linalg.eigh(covariance)
        front = eigenvectors[:, 0]
        front = front - up * np.dot(front, up)
    front = _normalise(front)
    right = _normalise(np.cross(front, up))
    return right, up, front


def _project(points: np.ndarray, horizontal: np.ndarray,
             vertical: np.ndarray) -> np.ndarray:
    return np.column_stack((points @ horizontal, points @ vertical))


def _sample_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, color="white", fontsize=12)
    axis.set_aspect("equal", adjustable="box")
    axis.set_facecolor("#101216")
    axis.tick_params(colors="#b7bcc5", labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#555b66")
    axis.set_xlabel("horizontal (m)", color="#b7bcc5")
    axis.set_ylabel("height (m)", color="#b7bcc5")


def _render_comparison(
        before: np.ndarray, after: np.ndarray, colours: np.ndarray,
        horizontal: np.ndarray, vertical: np.ndarray, view_name: str,
        output_path: Path, max_points: int) -> None:
    sample = _sample_indices(len(before), max_points)
    projected_before = _project(before[sample], horizontal, vertical)
    projected_after = _project(after[sample], horizontal, vertical)
    display_colours = colours[sample].astype(np.float64) / 255.0
    combined = np.vstack((projected_before, projected_after))
    padding = np.maximum(np.ptp(combined, axis=0) * 0.03, 0.01)
    limits = [(combined[:, axis].min() - padding[axis],
               combined[:, axis].max() + padding[axis]) for axis in range(2)]

    figure, axes = plt.subplots(1, 2, figsize=(14, 8), facecolor="#101216")
    for axis, projected, title in zip(
            axes, (projected_before, projected_after),
            ("Rigid reconstruction", "With non-rigid refinement")):
        axis.scatter(projected[:, 0], projected[:, 1], s=0.35,
                     c=display_colours, linewidths=0, alpha=0.78, rasterized=True)
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
        _style_axis(axis, f"{title} — {view_name}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _render_displacement(
        after: np.ndarray, displacement_m: np.ndarray, horizontal: np.ndarray,
        vertical: np.ndarray, output_path: Path, max_points: int) -> None:
    sample = _sample_indices(len(after), max_points)
    projected = _project(after[sample], horizontal, vertical)
    values_mm = displacement_m[sample] * 1000.0
    figure, axis = plt.subplots(figsize=(8, 8), facecolor="#101216")
    scatter = axis.scatter(projected[:, 0], projected[:, 1], s=0.45,
                           c=values_mm, cmap="magma", linewidths=0,
                           vmin=0.0, vmax=max(1.0, float(np.percentile(values_mm, 99))),
                           rasterized=True)
    _style_axis(axis, "Non-rigid movement — front view")
    colour_bar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    colour_bar.set_label("movement (mm)", color="white")
    colour_bar.ax.tick_params(colors="#b7bcc5")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _render_mobility(
        points: np.ndarray, mobility: np.ndarray, horizontal: np.ndarray,
        vertical: np.ndarray, output_path: Path, max_points: int) -> None:
    sample = _sample_indices(len(points), max_points)
    projected = _project(points[sample], horizontal, vertical)
    figure, axis = plt.subplots(figsize=(8, 8), facecolor="#101216")
    scatter = axis.scatter(
        projected[:, 0], projected[:, 1], s=0.45, c=mobility[sample],
        cmap="viridis", linewidths=0, vmin=0.0, vmax=1.0, rasterized=True)
    _style_axis(axis, "Allowed branch-tip mobility — front view")
    colour_bar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    colour_bar.set_label("mobility (0 = fixed, 1 = free)", color="white")
    colour_bar.ax.tick_params(colors="#b7bcc5")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)



def run() -> dict[str, Path]:
    input_path = INPUT_PLY.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input point cloud not found: {input_path}")
    poses_path = (POSES_PATH or _default_poses_path(input_path)).resolve()
    output_dir = (OUTPUT_DIR or input_path.parent / "non_rigid_comparison").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        work_voxel_m=WORK_VOXEL_M,
        node_spacing_m=NODE_SPACING_M,
        blend_nodes=BLEND_NODES,
        graph_neighbors=GRAPH_NEIGHBORS,
        static_support_radius_m=STATIC_SUPPORT_RADIUS_M,
        static_support_frames=STATIC_SUPPORT_FRAMES,
        shape_radius_m=SHAPE_RADIUS_M,
        linearity_start=LINEARITY_START,
        linearity_full=LINEARITY_FULL,
        density_full_below=DENSITY_FULL_BELOW,
        density_zero_above=DENSITY_ZERO_ABOVE,
        correspondence_radii_m=CORRESPONDENCE_RADII_M,
        min_consensus_frames=MIN_CONSENSUS_FRAMES,
        max_consensus_spread_fraction=MAX_CONSENSUS_SPREAD_FRACTION,
        min_point_mobility=MIN_POINT_MOBILITY,
        data_weight=DATA_WEIGHT,
        smoothness=SMOOTHNESS,
        tip_magnitude=TIP_MAGNITUDE,
        static_anchor=STATIC_ANCHOR,
        bottom_anchor_fraction=BOTTOM_ANCHOR_FRACTION,
        max_tip_displacement_m=MAX_TIP_DISPLACEMENT_M)

    cloud = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(cloud.points, dtype=np.float64)
    colours_float = np.asarray(cloud.colors, dtype=np.float64)
    if colours_float.shape != points.shape:
        raise ValueError("input must contain the per-frame colours written by reconstruction")
    up, frame_ids, final_poses = _load_up_and_frame_ids(poses_path)
    blocks = _split_frame_colours(points, colours_float, frame_ids)
    frames = _prepare_frames(blocks, up, settings)
    print(f"loaded {len(points):,} points from {len(frames)} frames")
    print("control nodes (movable) per frame:", ", ".join(
        f"{frame.frame_id}={len(frame.nodes)} "
        f"({np.count_nonzero(frame.node_mobility > settings.min_point_mobility)})"
        for frame in frames))

    before_work = [frame.work_original for frame in frames]
    mobile_masks = [
        np.asarray(frame.blend_work @ frame.node_mobility).ravel()
        > settings.min_point_mobility
        for frame in frames]
    metric_threshold = settings.correspondence_radii_m[-1]
    metrics_before = _alignment_metrics(before_work, metric_threshold)
    mobile_metrics_before = _alignment_metrics(
        before_work, metric_threshold, mobile_masks)
    history = optimise(frames, settings)
    after_work = [frame.work_deformed() for frame in frames]
    metrics_after = _alignment_metrics(after_work, metric_threshold)
    mobile_metrics_after = _alignment_metrics(
        after_work, metric_threshold, mobile_masks)

    before = np.concatenate([frame.original for frame in frames], axis=0)
    after_groups = [frame.full_deformed() for frame in frames]
    after = np.concatenate(after_groups, axis=0)
    colours = np.concatenate([
        np.repeat(frame.rgb[None, :], len(frame.original), axis=0) for frame in frames], axis=0)
    displacement_vectors = after - before
    displacement_m = np.linalg.norm(displacement_vectors, axis=1)
    point_mobility = np.concatenate([
        np.asarray(frame.blend_full @ frame.node_mobility).ravel()
        for frame in frames])

    stem = input_path.stem
    output_ply = output_dir / f"{stem}_nonrigid.ply"
    output_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(after))
    output_cloud.colors = o3d.utility.Vector3dVector(colours.astype(float) / 255.0)
    if not o3d.io.write_point_cloud(str(output_ply), output_cloud):
        raise OSError(f"failed to write {output_ply}")

    displacement_path = output_dir / f"{stem}_nonrigid_displacements.npz"
    np.savez_compressed(
        displacement_path,
        frame_ids=np.asarray(frame_ids),
        displacement_vectors_m=displacement_vectors,
        displacement_m=displacement_m,
        branch_tip_mobility=point_mobility)

    right, vertical, front = _view_basis(before, up, final_poses)
    front_comparison = output_dir / f"{stem}_comparison_front.png"
    side_comparison = output_dir / f"{stem}_comparison_side.png"
    movement_plot = output_dir / f"{stem}_movement_front.png"
    mobility_plot = output_dir / f"{stem}_mobility_front.png"
    _render_comparison(before, after, colours, right, vertical, "front view",
                       front_comparison, MAX_RENDER_POINTS)
    _render_comparison(before, after, colours, front, vertical, "side view",
                       side_comparison, MAX_RENDER_POINTS)
    _render_displacement(after, displacement_m, right, vertical, movement_plot,
                         MAX_RENDER_POINTS)
    _render_mobility(before, point_mobility, right, vertical, mobility_plot,
                     MAX_RENDER_POINTS)

    report = {
        "input_ply": str(input_path),
        "poses": str(poses_path),
        "settings": settings.__dict__,
        "point_count": len(points),
        "frame_point_counts": {frame.frame_id: len(frame.original) for frame in frames},
        "control_node_counts": {frame.frame_id: len(frame.nodes) for frame in frames},
        "before": metrics_before,
        "after": metrics_after,
        "movable_branch_points_before": mobile_metrics_before,
        "movable_branch_points_after": mobile_metrics_after,
        "mobility": {
            "movable_point_fraction": float(np.mean(
                point_mobility > settings.min_point_mobility)),
            "fully_free_point_fraction": float(np.mean(point_mobility >= 0.95)),
        },
        "movement": {
            "median_m": float(np.median(displacement_m)),
            "p90_m": float(np.percentile(displacement_m, 90)),
            "p99_m": float(np.percentile(displacement_m, 99)),
            "max_m": float(displacement_m.max()),
        },
        "optimisation_history": history,
    }
    report_path = output_dir / f"{stem}_nonrigid_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    before_mm = metrics_before["median_cross_frame_distance_m"] * 1000.0
    after_mm = metrics_after["median_cross_frame_distance_m"] * 1000.0
    mobile_before_mm = mobile_metrics_before["p90_cross_frame_distance_m"] * 1000.0
    mobile_after_mm = mobile_metrics_after["p90_cross_frame_distance_m"] * 1000.0
    print(f"global median cross-frame distance: {before_mm:.1f} -> {after_mm:.1f} mm")
    print(f"movable branch-point p90 distance: "
          f"{mobile_before_mm:.1f} -> {mobile_after_mm:.1f} mm")
    print(f"point movement: p90 {np.percentile(displacement_m, 90) * 1000.0:.1f} mm, "
          f"p99 {np.percentile(displacement_m, 99) * 1000.0:.1f} mm")
    print(f"outputs -> {output_dir}")
    return {
        "point_cloud": output_ply,
        "displacements": displacement_path,
        "report": report_path,
        "front_comparison": front_comparison,
        "side_comparison": side_comparison,
        "movement_plot": movement_plot,
        "mobility_plot": mobility_plot,
    }


def main() -> None:
    run()


if __name__ == "__main__":
    main()
