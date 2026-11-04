"""Conservative non-rigid refinement of already rigidly registered frame clouds.

Only sparse, thin branch ends with weak cross-frame support are allowed to move.
Per-frame deformation-graph nodes are fitted to a frame-balanced median consensus;
stable geometry and the lower part of the scene remain strongly anchored.
"""

from dataclasses import dataclass
import warnings

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix, csr_matrix, diags, vstack
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree


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
    original: np.ndarray
    colors: np.ndarray
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

    def full_deformed(self):
        return self.original + self.blend_full @ self.node_displacement

    def work_deformed(self):
        return self.work_original + self.blend_work @ self.node_displacement


def settings_from_config(cfg) -> Settings:
    """Build typed settings from the ``non_rigid`` configuration namespace."""
    return Settings(
        work_voxel_m=cfg.work_voxel_m,
        node_spacing_m=cfg.node_spacing_m,
        blend_nodes=cfg.blend_nodes,
        graph_neighbors=cfg.graph_neighbors,
        static_support_radius_m=cfg.static_support_radius_m,
        static_support_frames=cfg.static_support_frames,
        shape_radius_m=cfg.shape_radius_m,
        linearity_start=cfg.linearity_start,
        linearity_full=cfg.linearity_full,
        density_full_below=cfg.density_full_below,
        density_zero_above=cfg.density_zero_above,
        correspondence_radii_m=tuple(cfg.correspondence_radii_m),
        min_consensus_frames=cfg.min_consensus_frames,
        max_consensus_spread_fraction=cfg.max_consensus_spread_fraction,
        min_point_mobility=cfg.min_point_mobility,
        data_weight=cfg.data_weight,
        smoothness=cfg.smoothness,
        tip_magnitude=cfg.tip_magnitude,
        static_anchor=cfg.static_anchor,
        bottom_anchor_fraction=cfg.bottom_anchor_fraction,
        max_tip_displacement_m=cfg.max_tip_displacement_m)


def _voxel_representatives(points, voxel_m):
    origin = points.min(axis=0)
    cells = np.floor((points - origin) / voxel_m).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)]


def _blend_matrix(points, nodes, count, spacing_m):
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


def _smoothness_matrix(nodes, neighbor_count):
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
    rows, cols, values = [], [], []
    for row, (first, second) in enumerate(sorted(edges)):
        rows.extend((row, row))
        cols.extend((first, second))
        values.extend((1.0, -1.0))
    return coo_matrix(
        (values, (rows, cols)), shape=(len(edges), len(nodes))).tocsr()


def _camera_up(final_poses, frame_ids):
    camera_down = np.array([
        np.asarray(final_poses[fid])[:3, :3] @ np.array([0.0, 1.0, 0.0])
        for fid in frame_ids])
    up = -camera_down.mean(axis=0)
    return up / (np.linalg.norm(up) + 1e-12)


def _local_shape_scores(frame, radius_m):
    neighborhoods = cKDTree(frame.work_original).query_ball_point(frame.nodes, radius_m)
    density = np.asarray([len(indices) for indices in neighborhoods], dtype=np.int32)
    linearity = np.zeros(len(frame.nodes), dtype=np.float64)
    for index, neighbors in enumerate(neighborhoods):
        if len(neighbors) < 8:
            continue
        local = frame.work_original[neighbors]
        covariance = np.cov(local - local.mean(axis=0), rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
        linearity[index] = ((eigenvalues[0] - eigenvalues[1]) /
                            (eigenvalues[0] + 1e-12))
    return linearity, density


def _assign_branch_tip_mobility(frames, settings):
    work_trees = [cKDTree(frame.work_original) for frame in frames]
    linearity_width = max(settings.linearity_full - settings.linearity_start, 1e-6)
    density_width = max(settings.density_zero_above - settings.density_full_below, 1)
    for index, frame in enumerate(frames):
        support = np.zeros(len(frame.nodes), dtype=np.int32)
        for other, tree in enumerate(work_trees):
            if other != index:
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


def _prepare_frames(clouds, final_poses, settings):
    nonempty = [(fid, pcd) for fid, pcd in clouds.items() if len(pcd.points)]
    if len(nonempty) < 2:
        raise ValueError("non-rigid refinement needs at least two non-empty frame clouds")
    all_points = np.concatenate([np.asarray(p.points) for _, p in nonempty], axis=0)
    up = _camera_up(final_poses, [fid for fid, _ in nonempty])
    heights = all_points @ up
    low, high = np.percentile(heights, (1.0, 99.0))
    height_range = max(float(high - low), 1e-6)
    frames = []
    for frame_id, cloud in nonempty:
        original = np.asarray(cloud.points, dtype=np.float64).copy()
        colors = np.asarray(cloud.colors, dtype=np.float64).copy()
        if colors.shape != original.shape:
            colors = np.zeros_like(original)
        work = _voxel_representatives(original, settings.work_voxel_m)
        nodes = _voxel_representatives(work, settings.node_spacing_m)
        node_height = np.clip((nodes @ up - low) / height_range, 0.0, 1.0)
        count = len(nodes)
        frames.append(FrameCloud(
            frame_id=frame_id,
            original=original,
            colors=colors,
            work_original=work,
            nodes=nodes,
            blend_full=_blend_matrix(
                original, nodes, settings.blend_nodes, settings.node_spacing_m),
            blend_work=_blend_matrix(
                work, nodes, settings.blend_nodes, settings.node_spacing_m),
            smooth_matrix=_smoothness_matrix(nodes, settings.graph_neighbors),
            node_height=node_height,
            node_support=np.zeros(count, dtype=np.int32),
            node_linearity=np.zeros(count),
            node_density=np.zeros(count, dtype=np.int32),
            node_mobility=np.zeros(count),
            node_displacement=np.zeros((count, 3))))
    _assign_branch_tip_mobility(frames, settings)
    return frames


def _consensus_correspondences(source, target_groups, radius_m, point_mobility,
                               settings):
    distances, candidates = [], []
    for target in target_groups:
        distance, indices = cKDTree(target).query(source, k=1)
        distances.append(distance)
        candidates.append(target[indices])
    distances = np.stack(distances, axis=1)
    candidates = np.stack(candidates, axis=1)
    inside = distances < radius_m
    masked = np.where(inside[:, :, None], candidates, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        consensus = np.nanmedian(masked, axis=1)
        spread = np.nanmedian(np.where(
            inside, np.linalg.norm(candidates - consensus[:, None, :], axis=2), np.nan),
            axis=1)
    support_count = inside.sum(axis=1)
    required = min(settings.min_consensus_frames, len(target_groups))
    valid = np.flatnonzero(
        (support_count >= required)
        & (point_mobility > settings.min_point_mobility)
        & np.isfinite(consensus[:, 0])
        & (spread < radius_m * settings.max_consensus_spread_fraction))
    if not len(valid):
        return valid, consensus[valid], np.empty(0), np.empty(0)
    support_weight = np.minimum(support_count[valid], 5) / min(len(target_groups), 5)
    coherence_weight = np.exp(
        -0.5 * (spread[valid] / max(0.5 * radius_m, 1e-6)) ** 2)
    weights = settings.data_weight * (0.1 + 0.9 * point_mobility[valid])
    weights *= support_weight * coherence_weight
    residual = np.linalg.norm(consensus[valid] - source[valid], axis=1)
    return valid, consensus[valid], weights, residual


def _solve_frame(frame, target_groups, radius_m, settings):
    source_current = frame.work_deformed()
    point_mobility = np.asarray(frame.blend_work @ frame.node_mobility).ravel()
    indices, consensus, weights, residuals = _consensus_correspondences(
        source_current, target_groups, radius_m, point_mobility, settings)
    if len(indices) < max(30, len(frame.nodes) // 4):
        return frame.node_displacement.copy(), {
            "matches": int(len(indices)),
            "median_consensus_residual_m": (
                float(np.median(residuals)) if len(residuals) else None)}

    weighted_data = diags(np.sqrt(weights)) @ frame.blend_work[indices]
    target = consensus - frame.work_original[indices]
    weighted_target = np.sqrt(weights)[:, None] * target
    magnitude = (settings.tip_magnitude
                 + settings.static_anchor * (1.0 - frame.node_mobility) ** 2)
    regularise = diags(np.sqrt(magnitude))
    smooth = np.sqrt(settings.smoothness) * frame.smooth_matrix
    system = vstack((weighted_data, smooth, regularise), format="csr")
    rhs = np.vstack((weighted_target,
                     np.zeros((smooth.shape[0] + regularise.shape[0], 3))))
    displacement = np.empty((len(frame.nodes), 3), dtype=np.float64)
    for axis in range(3):
        displacement[:, axis] = lsqr(
            system, rhs[:, axis], atol=1e-6, btol=1e-6, iter_lim=250)[0]
    lengths = np.linalg.norm(displacement, axis=1)
    limits = settings.max_tip_displacement_m * frame.node_mobility
    over = lengths > limits
    displacement[over] *= (limits[over] / lengths[over])[:, None]
    return displacement, {
        "matches": int(len(indices)),
        "median_consensus_residual_m": float(np.median(residuals)),
        "max_node_displacement_m": float(np.linalg.norm(displacement, axis=1).max())}


def _optimise(frames, settings):
    history = []
    for pass_index, radius_m in enumerate(settings.correspondence_radii_m, 1):
        snapshots = [frame.work_deformed() for frame in frames]
        updates, frame_stats = [], {}
        for index, frame in enumerate(frames):
            targets = [cloud for other, cloud in enumerate(snapshots) if other != index]
            update, stats = _solve_frame(frame, targets, radius_m, settings)
            updates.append(update)
            frame_stats[frame.frame_id] = stats
        for frame, update in zip(frames, updates):
            frame.node_displacement = update
        history.append({"pass": pass_index,
                        "correspondence_radius_m": radius_m,
                        "frames": frame_stats})
        finite = [s["median_consensus_residual_m"] for s in frame_stats.values()
                  if s["median_consensus_residual_m"] is not None]
        median_mm = 1000.0 * float(np.median(finite)) if finite else float("nan")
        print(f"non-rigid pass {pass_index}/{len(settings.correspondence_radii_m)} "
              f"(r={radius_m * 1000:.0f}mm): median residual={median_mm:.1f}mm")
    return history


def refine_clouds(clouds, final_poses, settings):
    """Return deformed copies of rigidly registered per-frame clouds and metadata."""
    frames = _prepare_frames(clouds, final_poses, settings)
    history = _optimise(frames, settings)
    refined, displacements, mobility = {}, {}, {}
    frame_map = {frame.frame_id: frame for frame in frames}
    for frame_id, original_cloud in clouds.items():
        if frame_id not in frame_map:
            refined[frame_id] = o3d.geometry.PointCloud(original_cloud)
            displacements[frame_id] = np.empty((0, 3))
            mobility[frame_id] = np.empty(0)
            continue
        frame = frame_map[frame_id]
        points = frame.full_deformed()
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.colors = o3d.utility.Vector3dVector(frame.colors)
        refined[frame_id] = cloud
        displacements[frame_id] = points - frame.original
        mobility[frame_id] = np.asarray(
            frame.blend_full @ frame.node_mobility).ravel()
    return refined, {"history": history,
                     "displacements": displacements,
                     "mobility": mobility}
