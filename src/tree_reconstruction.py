"""Reconstruct a tree capture into one merged point cloud.

Detect the calibration marker in every depth+intensity frame with SAM 3, chain
the marker poses to give each frame an initial global pose, refine them with
pairwise point-to-plane ICP and an anchored pose graph, then merge the frames,
isolate the target tree, and filter the cloud before saving. Falls back to
marker-only alignment when too few good ICP constraints survive.

Transform notation: T_A_B maps points from frame A into frame B
(p_B = T_A_B @ p_A). detect_marker returns T_marker_cam (marker pose in the
frame's camera); chaining through the shared marker gives each frame's initial
global pose in the reference camera's frame:
    T_cam_i_cam_ref = T_marker_cam_ref @ inv(T_marker_cam_i)
"""

import glob
import os

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d, minimum_filter
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

import config
import depth_regions as dr
import marker_detection_sam3 as m
import marker_registration as reg
import viz_utils as vz

# Pipeline knobs from cfg/default.yaml (see config.py).
_PG = config.CFG.pose_graph
_TI = config.CFG.tree_isolation
_SF = config.CFG.support_filter
_VF = config.CFG.visibility_filter

# pairwise point-to-plane ICP (pose-graph edges)
ICP_VOXEL = _PG.icp_voxel               # downsample resolution (m)
ICP_NORMAL_RADIUS = ICP_VOXEL * 2.5
ICP_DIST_COARSE = ICP_VOXEL * 5.0       # coarse max correspondence distance (m)
ICP_DIST_FINE = ICP_VOXEL * 1.5         # fine max correspondence distance (m)
ICP_MAX_ITER = _PG.icp_max_iter
ICP_TUKEY_K = _PG.icp_tukey_k_m         # robust kernel width (m); 0 disables

# isolate the target tree (drop neighboring trees)
TREE_ISOLATION_ENABLED = _TI.enabled
CLUSTER_EPS = _TI.cluster_eps_m               # DBSCAN neighbor distance (m)
CLUSTER_MIN_POINTS = _TI.cluster_min_points   # min points for a cluster
MIN_CLUSTER_FRAC = _TI.min_cluster_frac       # ignore clusters below this fraction of the largest
SPLIT_FUSED_TREES = _TI.split_fused_trees     # split a neighbor tree fused into the selected cluster
SPLIT_GAP_FRAC = _TI.split_gap_frac           # valley is a between-tree gap only if below this * smaller flanking peak
SPLIT_MIN_LOBE_FRAC = _TI.split_min_lobe_frac  # both kept + dropped parts must exceed this fraction, else one tree
SPLIT_BIN_M = 0.03                            # along-row density histogram bin (m)
SPLIT_SMOOTH = 1.5                            # gaussian smoothing (bins)
SPLIT_PROM_FRAC = 0.08                        # min valley prominence as a fraction of peak density
SPLIT_MIN_SPAN_M = 0.4                        # a cluster narrower than this can't hold two trees

# multi-view support filter on the merged cloud
SUPPORT_ENABLED = _SF.enabled
SUPPORT_RADIUS = _SF.radius_m           # support ball around each merged point (m)
SUPPORT_MIN_FRAMES = _SF.min_frames     # distinct frames required inside the ball

# visibility (free-space) filter on the merged cloud
VIS_ENABLED = _VF.enabled
VIS_TOL_M = _VF.tol_m                            # free-space margin in front of the observed surface (m)
VIS_MIN_CONTRADICTIONS = _VF.min_contradictions  # frames that must see through a point before it is dropped
VIS_MIN_CONFIRMATIONS = _VF.min_confirmations    # frames that must observe a point to keep it regardless
VIS_CONFIRM_RADIUS = _VF.confirm_radius_m        # 3D ball for counting confirming frames (m)
VIS_PIXEL_MARGIN = _VF.pixel_margin              # +/- window for the nearest observed depth; 0 = exact pixel
VIS_MIN_DEPTH_M = _VF.min_depth_m                # ignore points closer than this to a camera (m)

# which frame pairs to connect, and which edges to trust
PAIR_NEIGHBOR_SPAN = _PG.pair_neighbor_span   # add edges between frames with |i-j| <= this
EDGE_MIN_FITNESS = _PG.edge_min_fitness       # drop pairwise constraints below this fitness
EDGE_MAX_RMSE_M = _PG.edge_max_rmse_m         # ...or above this inlier RMSE

# global pose-graph optimization
POSE_GRAPH_PRUNE_THRESHOLD = _PG.prune_threshold  # optimizer's edge-confidence prune


def discover_frame_ids(frame_dir):
    """Sorted frame ids (the *_depth.png basenames) in `frame_dir`, numerically
    if all ids parse as int, else lexicographically."""
    paths = glob.glob(os.path.join(frame_dir, "*_depth.png"))
    ids = [os.path.basename(p)[:-len("_depth.png")] for p in paths]
    try:
        ids.sort(key=int)
    except ValueError:
        ids.sort()
    return ids


def invert_rigid_transform(T):
    """Inverse of a rigid SE(3) transform via R.T / -R.T@t."""
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def detect_all_markers(frame_dir, frame_ids, visualize=False):
    """Run detect_marker on every frame. Returns an insertion-ordered dict
    frame_id -> {T_marker_cam, depth_m, intensity, K} for accepted frames only."""
    detections = {}
    for frame_id in frame_ids:
        intensity, intensity8, depth_m, K = dr.load_frame(frame_dir, frame_id)
        T_marker_cam = m.detect_marker(
            depth_m, intensity, intensity8, K, visualize=visualize,
            frame_dir=frame_dir, frame_id=frame_id)
        if T_marker_cam is None:
            print(f"[frame {frame_id}] no accepted marker -- skipped")
            continue
        print(f"[frame {frame_id}] marker accepted")
        detections[frame_id] = dict(T_marker_cam=T_marker_cam, depth_m=depth_m,
                                    intensity=intensity, K=K)
    return detections


def build_initial_global_poses(detections):
    """Marker-based initial global pose (camera frame -> reference) per frame. The
    reference is the first accepted detection (identity). Returns
    (ref_id, {frame_id: T_cam_i_cam_ref})."""
    ref_id = next(iter(detections))
    T_marker_cam_ref = detections[ref_id]["T_marker_cam"]
    poses = {}
    for frame_id, d in detections.items():
        if frame_id == ref_id:
            poses[frame_id] = np.eye(4)
        else:
            poses[frame_id] = T_marker_cam_ref @ invert_rigid_transform(d["T_marker_cam"])
    return ref_id, poses


def make_initial_clouds(detections):
    """Full-resolution point cloud per frame, in its own camera frame, each
    painted a distinct color. Global placement is carried by the pose graph. 
    Returns {frame_id: PointCloud}."""
    clouds = {}
    palette = vz.distinct_colors(len(detections))
    for color, (frame_id, d) in zip(palette, detections.items()):
        pts, _ = reg.lift_to_3d((d["depth_m"] > 0).astype(np.uint8) * 255,
                                d["depth_m"], d["K"], d["intensity"])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.paint_uniform_color(color)
        clouds[frame_id] = pcd
    return clouds


def preprocess_for_icp(clouds):
    """Downsample + estimate normals once per frame (point-to-plane ICP needs
    normals). Returns {frame_id: downsampled PointCloud with normals}."""
    down = {}
    for frame_id, pcd in clouds.items():
        d = pcd.voxel_down_sample(ICP_VOXEL)
        d.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=ICP_NORMAL_RADIUS, max_nn=30))
        down[frame_id] = d
    return down


def pairwise_icp_registration(source, target, init_source_to_target):
    """Point-to-plane ICP refining the source->target relative pose from the
    marker init: a plain coarse pass, then a fine pass with a Tukey kernel (width
    ICP_TUKEY_K) so geometry that moved between frames gets zero weight and only
    static structure drives the pose. Returns (T_source_target, information,
    fitness, rmse); `source`/`target` are downsampled clouds with normals."""
    fine_estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.TukeyLoss(k=ICP_TUKEY_K)) if ICP_TUKEY_K > 0 \
        else o3d.pipelines.registration.TransformationEstimationPointToPlane()
    coarse = o3d.pipelines.registration.registration_icp(
        source, target, ICP_DIST_COARSE, init_source_to_target,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER))
    fine = o3d.pipelines.registration.registration_icp(
        source, target, ICP_DIST_FINE, coarse.transformation,
        fine_estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER))
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, ICP_DIST_FINE, fine.transformation)
    return fine.transformation, information, float(fine.fitness), float(fine.inlier_rmse)


def build_pose_graph(frame_ids, poses, down_clouds):
    """Pose graph whose nodes are the frames (initial global poses) and whose
    edges are accepted pairwise ICP constraints between frames within
    PAIR_NEIGHBOR_SPAN. Consecutive frames give "certain" edges, farther pairs
    "uncertain" (loop-closure) edges; node 0 is the reference. Returns
    (pose_graph, n_edges)."""
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for fid in frame_ids:
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(poses[fid]))

    n_edges = 0
    for i in range(len(frame_ids)):
        for j in range(i + 1, len(frame_ids)):
            if j - i > PAIR_NEIGHBOR_SPAN:
                continue
            src_id, tgt_id = frame_ids[i], frame_ids[j]
            # source->target init from the global init poses: T = inv(pose_tgt) @ pose_src
            init = invert_rigid_transform(poses[tgt_id]) @ poses[src_id]
            T, info, fitness, rmse = pairwise_icp_registration(
                down_clouds[src_id], down_clouds[tgt_id], init)

            if fitness < EDGE_MIN_FITNESS or rmse > EDGE_MAX_RMSE_M:
                print(f"  edge {src_id}->{tgt_id}: SKIP "
                      f"(fitness={fitness:.3f} rmse={rmse * 1000:.1f}mm)")
                continue

            uncertain = (j != i + 1)
            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                i, j, T, info, uncertain=uncertain))
            n_edges += 1
            print(f"  edge {src_id}->{tgt_id}: "
                  f"fitness={fitness:.3f} rmse={rmse * 1000:.1f}mm"
                  f"{' (loop)' if uncertain else ''}")
    return pose_graph, n_edges


def optimize_pose_graph(pose_graph):
    """Jointly refine all node poses so every edge is respected at once, keeping
    node 0 (the reference) fixed. Modifies + returns pose_graph."""
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=ICP_DIST_FINE,
        edge_prune_threshold=POSE_GRAPH_PRUNE_THRESHOLD,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
    return pose_graph


def _transform_clouds(frame_ids, clouds, pose_list):
    """Transform each full-res cloud in place by its pose (order matches
    frame_ids). Returns the same {frame_id: PointCloud} dict."""
    for fid, pose in zip(frame_ids, pose_list):
        clouds[fid].transform(pose)
    return clouds


def apply_optimized_poses(frame_ids, clouds, pose_graph):
    """Place each full-res cloud into the reference frame using its optimized
    pose-graph node pose."""
    poses = [pose_graph.nodes[i].pose for i in range(len(frame_ids))]
    return _transform_clouds(frame_ids, clouds, poses)


def register_to_reference(detections):
    """Bring every frame's cloud into the reference frame (first accepted
    detection): marker chaining initialises each pose, pairwise ICP builds
    relative-pose constraints, and an anchored pose graph jointly optimises them,
    falling back to marker-only placement when too few constraints survive.
    Returns (ref_frame_id, {frame_id: PointCloud}, {frame_id: T_cam_ref}), each
    cloud already in the reference frame and the final poses returned so the
    visibility filter can reproject back into each camera."""
    if not detections:
        raise ValueError("no frame had an accepted marker detection")

    ref_id, poses = build_initial_global_poses(detections)
    clouds = make_initial_clouds(detections)
    frame_ids = list(clouds)

    down_clouds = preprocess_for_icp(clouds)
    print("building pairwise ICP constraints:")
    pose_graph, n_edges = build_pose_graph(frame_ids, poses, down_clouds)

    if n_edges == 0:
        print("no usable pairwise constraints -- falling back to marker-only alignment")
        clouds = _transform_clouds(frame_ids, clouds, [poses[f] for f in frame_ids])
        return ref_id, clouds, poses

    try:
        optimize_pose_graph(pose_graph)
        final_poses = {fid: np.asarray(pose_graph.nodes[i].pose)
                       for i, fid in enumerate(frame_ids)}
        clouds = apply_optimized_poses(frame_ids, clouds, pose_graph)
        print(f"global optimization over {n_edges} edges done")
    except Exception as e:  # optimization is best-effort; keep marker alignment
        print(f"pose-graph optimization failed ({e}) -- falling back to marker-only")
        clouds = _transform_clouds(frame_ids, clouds, [poses[f] for f in frame_ids])
        final_poses = poses

    return ref_id, clouds, final_poses


def merge_clouds(clouds):
    """Combine the per-frame clouds (already in the reference frame) into one."""
    merged = o3d.geometry.PointCloud()
    for pcd in clouds.values():
        merged += pcd
    return merged


def _camera_up(final_poses, frame_ids):
    """World-up in the reference frame: mean over frames of each camera's negated
    +y axis (image y points down), averaged for robustness. Unit 3-vector."""
    downs = np.array([np.asarray(final_poses[f])[:3, :3] @ np.array([0., 1., 0.])
                      for f in frame_ids])
    up = -downs.mean(0)
    return up / (np.linalg.norm(up) + 1e-9)


def _along_row_density(P, up):
    """Smoothed 1-D point density of `P` along the row (its widest horizontal
    direction after dropping the `up` component). Returns (s, row, ctr, dens):
    per-point along-row coordinate, row axis, bin centres, smoothed counts. Two
    trees show two humps split by a valley; a lone tree, one hump."""
    hpc = P - np.outer(P @ up, up)
    hpc = hpc - hpc.mean(0)
    _, _, Vh = np.linalg.svd(hpc, full_matrices=False)
    row = Vh[0]                                    # widest horizontal direction = the row
    s = hpc @ row
    nb = max(8, int(np.ceil(np.ptp(s) / SPLIT_BIN_M)))
    hist, edges = np.histogram(s, bins=nb)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    dens = gaussian_filter1d(hist.astype(float), SPLIT_SMOOTH)
    return s, row, ctr, dens


def _row_gap_cuts(ctr, dens):
    """Along-row cut positions: density valleys deep enough (below SPLIT_GAP_FRAC
    of the smaller flanking peak) to be genuine between-tree gaps rather than
    ordinary canopy dips."""
    valleys, _ = find_peaks(-dens, prominence=SPLIT_PROM_FRAC * dens.max())
    return [ctr[v] for v in valleys
            if dens[v] <= SPLIT_GAP_FRAC * min(dens[:v].max() if v > 0 else np.inf,
                                               dens[v + 1:].max() if v < len(dens) - 1 else np.inf)]


def _split_fused_tree(P, cams, axes, up):
    """Boolean keep-mask over `P` splitting a neighbor tree fused into the target
    cluster: cut at every between-tree gap along the row (_row_gap_cuts) and keep
    the segment the cameras aim at most directly. All-True when there is no real
    second tree, so a single tree is never cut. `cams`/`axes` are the camera
    centres and unit optical axes; `up` is world-up in the reference frame."""
    s, row, ctr, dens = _along_row_density(P, up)
    if np.ptp(s) < SPLIT_MIN_SPAN_M:               # too narrow to hold two trees
        return np.ones(len(P), bool)

    cuts = _row_gap_cuts(ctr, dens)
    if not cuts:
        return np.ones(len(P), bool)

    bounds = np.concatenate([[-np.inf], sorted(cuts), [np.inf]])
    segs = [m for m in ((s > bounds[i]) & (s <= bounds[i + 1])
                        for i in range(len(bounds) - 1)) if m.any()]
    if len(segs) < 2:
        return np.ones(len(P), bool)

    def seg_look(mask):
        d = P[mask].mean(0) - cams
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        return float(np.degrees(np.arccos(np.clip((d * axes).sum(1), -1, 1))).mean())

    keep = min(segs, key=seg_look)                 # the camera-aimed tree
    if (~keep).sum() < SPLIT_MIN_LOBE_FRAC * len(P):  # dropped part too small -> one tree
        return np.ones(len(P), bool)
    return keep


def _isolation_geometries(pts, labels, big, target, cams, axes):
    """Open3D geometries for the cluster view: the cloud colored per substantial
    cluster (small clusters + noise gray), a coordinate frame + optical-axis arrow
    at each camera, and the selected target cluster in a green box."""
    colors = np.full((len(pts), 3), 0.55)         # noise / sub-threshold clusters: gray
    for col, c in zip(vz.distinct_colors(max(len(big), 1)), big):
        colors[labels == c] = col
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geoms = [vz.coord_frame(0.2), pcd]
    for c, a in zip(cams, axes):
        geoms.append(vz.coord_frame(0.06, origin=c))
        geoms.append(vz.normal_arrow(c, a, length=0.6, color=(0.05, 0.05, 0.05)))
    box = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(pts[labels == target])).get_axis_aligned_bounding_box()
    box.color = (0, 1, 0)
    geoms.append(box)
    return geoms


def _split_geometries(pts, labels, target, keep):
    """Open3D geometries for the split view: kept target green, dropped
    fused-neighbor points red, everything else light gray."""
    colors = np.full((len(pts), 3), 0.75)
    colors[(labels == target) & ~keep] = (0.9, 0.1, 0.1)   # dropped fused neighbor
    colors[keep] = (0.1, 0.8, 0.1)                          # kept target tree
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return [vz.coord_frame(0.2), pcd]


def _density_chart(P, up, kept, w=760, h=320):
    """cv2 bar chart of the along-row density used for the split: each bin tinted
    green where its points are kept, red where dropped, with a red line at every
    cut. Returns a titled BGR image."""
    s, row, ctr, dens = _along_row_density(P, up)
    cuts = _row_gap_cuts(ctr, dens)
    n = len(dens)
    binidx = np.clip(np.digitize(s, np.linspace(s.min(), s.max(), n + 1)) - 1, 0, n - 1)
    img = np.full((h, w, 3), 30, np.uint8)
    mx = dens.max() + 1e-9
    for i, d in enumerate(dens):
        x0, x1 = int(i * w / n), int((i + 1) * w / n)
        bh = int((h - 50) * d / mx)
        inb = binidx == i
        col = (90, 200, 90) if inb.any() and kept[inb].mean() >= 0.5 else (90, 90, 220)
        cv2.rectangle(img, (x0, h - 24 - bh), (max(x0, x1 - 1), h - 24), col, -1)
    for cpos in cuts:
        x = int(np.clip((cpos - s.min()) / (np.ptp(s) + 1e-9), 0, 1) * w)
        cv2.line(img, (x, 8), (x, h - 24), (60, 60, 240), 2)
    return vz.put_title(img, f"along-row density  (green=kept  red=dropped  cuts={len(cuts)})")


def _show_isolation(pts, labels, big, target, cams, axes, keep, up):
    """Pop up the three tree-isolation views, each blocking until closed: clusters
    + camera aim with the target boxed, kept target vs dropped fused neighbor, and
    the along-row density chart showing where the split cut."""
    o3d.visualization.draw_geometries(
        _isolation_geometries(pts, labels, big, target, cams, axes),
        window_name="tree isolation 1/3: clusters (colored) + camera aim; target boxed green")
    o3d.visualization.draw_geometries(
        _split_geometries(pts, labels, target, keep),
        window_name="tree isolation 2/3: kept target (green) vs dropped fused neighbor (red)")
    P = pts[labels == target]
    vz.show_scrollable("tree isolation 3/3: along-row density",
                       _density_chart(P, up, keep[labels == target]))


def isolate_target_tree(clouds, final_poses, visualize=False):
    """Keep only the target tree, the one the cameras were aimed at and drop
    neighboring trees. Split the merged cloud into DBSCAN clusters and keep the one the optical
    axes point most directly at (smallest mean look-angle). A neighbor physically
    fused into that cluster is then split off along the row (_split_fused_tree).
    `final_poses` gives each camera's centre (translation) and optical axis
    (R @ +z). Returns a new {frame_id: PointCloud} (input unchanged if no cluster
    is found), colors preserved."""
    per_frame_pts = [np.asarray(p.points) for p in clouds.values()]
    counts = [len(p) for p in per_frame_pts]
    if sum(counts) == 0:
        return clouds
    pts = np.vstack(per_frame_pts)

    merged = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    labels = np.asarray(merged.cluster_dbscan(
        eps=CLUSTER_EPS, min_points=CLUSTER_MIN_POINTS, print_progress=False))
    n_clusters = int(labels.max()) + 1
    if n_clusters <= 0:                      # everything came back as noise
        print("  tree isolation: no clusters found -- keeping all points")
        return clouds

    sizes = np.array([int((labels == c).sum()) for c in range(n_clusters)])
    big = np.where(sizes >= max(CLUSTER_MIN_POINTS, MIN_CLUSTER_FRAC * sizes.max()))[0]

    # camera centres + unit optical axes in the reference frame
    cams = np.array([np.asarray(final_poses[fid])[:3, 3] for fid in clouds])
    axes = np.array([np.asarray(final_poses[fid])[:3, :3] @ np.array([0., 0., 1.])
                     for fid in clouds])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)

    def look_angle(centroid):
        """Mean angle (deg) over cameras between each optical axis and the
        direction to `centroid` -- small = cameras aim at it."""
        d = centroid - cams
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        return float(np.degrees(np.arccos(np.clip((d * axes).sum(1), -1, 1))).mean())

    angles = {int(c): look_angle(pts[labels == c].mean(0)) for c in big}
    target = min(angles, key=angles.get)
    keep = labels == target

    # split off a neighbor tree fused into the selected cluster, keeping the
    # camera-aimed side; a lone tree is unimodal and comes back unchanged.
    split_dropped = 0
    up = None
    if SPLIT_FUSED_TREES and keep.sum() > 0:
        idx_keep = np.where(keep)[0]
        up = _camera_up(final_poses, clouds)
        sub = _split_fused_tree(pts[keep], cams, axes, up)
        split_dropped = int((~sub).sum())
        keep[idx_keep[~sub]] = False

    print(f"  tree isolation: {n_clusters} clusters ({len(big)} substantial); kept "
          f"cluster {target} ({int(keep.sum())} pts, cameras {angles[target]:.0f} deg "
          f"off its centre) of {len(pts)} pts -- dropped {len(pts) - int(keep.sum())} "
          f"pts ({split_dropped} a fused neighbor tree) from other trees/noise")

    if visualize:
        _show_isolation(pts, labels, big, target, cams, axes, keep,
                        up if up is not None else _camera_up(final_poses, clouds))

    out, idx = {}, 0
    for (fid, pcd), n in zip(clouds.items(), counts):
        m = keep[idx:idx + n]
        idx += n
        sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pcd.points)[m]))
        sub.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[m])
        out[fid] = sub
    return out


def filter_by_frame_support(clouds):
    """Multi-view support filter over the per-frame clouds (already in the
    reference frame). Static structure lands in the same place in every frame that
    sees it; a ghost of a moved branch or a single-frame speckle is not. Keep a
    point iff points from >= SUPPORT_MIN_FRAMES distinct frames fall within
    SUPPORT_RADIUS of it. Returns (kept, removed) merged clouds, colors preserved."""
    per_frame_pts = [np.asarray(p.points) for p in clouds.values()]
    merged_pts = np.vstack(per_frame_pts)
    merged_colors = np.vstack([np.asarray(p.colors) for p in clouds.values()])

    support = np.zeros(len(merged_pts), dtype=np.int32)
    for pts in per_frame_pts:
        if len(pts) == 0:
            continue
        dist, _ = cKDTree(pts).query(merged_pts, k=1,
                                     distance_upper_bound=SUPPORT_RADIUS,
                                     workers=-1)
        support += np.isfinite(dist)   # inf when no point of this frame is in range
    keep = support >= SUPPORT_MIN_FRAMES

    def subcloud(mask):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged_pts[mask]))
        pcd.colors = o3d.utility.Vector3dVector(merged_colors[mask])
        return pcd

    return subcloud(keep), subcloud(~keep)


def _nearest_observed_depth(depth, margin):
    """Per-pixel nearest valid observed depth within a +/-margin window; invalid
    pixels and empty windows are +inf. This min-pooling keeps the free-space test
    conservative near depth edges and thin structures. margin 0 = exact pixel."""
    d = np.where(np.isfinite(depth) & (depth > 0), depth, np.inf)
    if margin > 0:
        d = minimum_filter(d, size=2 * margin + 1, mode="nearest")
    return d


def filter_by_visibility(clouds, detections, final_poses):
    """Free-space (visibility) filter with a multi-view confirmation veto, over the
    per-frame clouds (already in the reference frame). Per merged point, count
    confirmations (frames with a point within VIS_CONFIRM_RADIUS in 3D, tolerating
    registration slop) and contradictions (frames whose observed depth along the
    ray sits more than VIS_TOL_M in front of the point -- they saw straight past
    it; occluded views are neutral). Keep a point iff confirmed by
    >= VIS_MIN_CONFIRMATIONS frames OR contradicted by < VIS_MIN_CONTRADICTIONS, so
    a branch two views truly captured survives however many others look through it.
    Returns (kept, removed) merged clouds, colors preserved."""
    per_frame_pts = [np.asarray(p.points) for p in clouds.values()]
    pts = np.vstack(per_frame_pts)
    colors = np.vstack([np.asarray(p.colors) for p in clouds.values()])

    # confirmations -- frames that observed a surface within VIS_CONFIRM_RADIUS (3D)
    confirmations = np.zeros(len(pts), dtype=np.int32)
    for fp in per_frame_pts:
        if len(fp) == 0:
            continue
        dist, _ = cKDTree(fp).query(pts, k=1, distance_upper_bound=VIS_CONFIRM_RADIUS,
                                    workers=-1)
        confirmations += np.isfinite(dist)   # inf when this frame has no point in range

    # contradictions -- frames that saw free space through the point
    contradictions = np.zeros(len(pts), dtype=np.int32)
    for fid, det in detections.items():
        K, depth = det["K"], det["depth_m"]
        H, W = depth.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        T_ref_cam = invert_rigid_transform(final_poses[fid])   # reference frame -> this camera
        cam = pts @ T_ref_cam[:3, :3].T + T_ref_cam[:3, 3]     # (N, 3) in camera coords
        z = cam[:, 2]

        in_front = z > VIS_MIN_DEPTH_M
        zc = np.where(in_front, z, 1.0)                        # guard the projection divide
        u = np.round(cam[:, 0] / zc * fx + cx).astype(np.int64)
        v = np.round(cam[:, 1] / zc * fy + cy).astype(np.int64)
        inside = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        observed = np.full(len(pts), np.inf)
        near = _nearest_observed_depth(depth, VIS_PIXEL_MARGIN)
        observed[inside] = near[v[inside], u[inside]]
        contradictions += inside & (z < observed - VIS_TOL_M)

    keep = (confirmations >= VIS_MIN_CONFIRMATIONS) | (contradictions < VIS_MIN_CONTRADICTIONS)

    def subcloud(mask):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[mask]))
        pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        return pcd

    return subcloud(keep), subcloud(~keep)


def output_path_for(frame_dir):
    """Destination .ply for `frame_dir` (.../<side>/<capture>/marker_captures/depth):
    <output_dir>/<side>/<capture>/<capture>_merged.ply."""
    capture_dir = os.path.dirname(os.path.dirname(os.path.abspath(frame_dir)))
    capture = os.path.basename(capture_dir)
    side = os.path.basename(os.path.dirname(capture_dir))
    return os.path.join(config.CFG.output_dir, side, capture, f"{capture}_merged.ply")


def save_frame_manifest(frame_dir, clouds, out_path):
    """Write a sidecar <capture>.frames.txt next to the .ply listing each
    contributing depth-frame path in reconstruction order (reference first); the
    contributing frames are exactly the keys of `clouds`. Returns the path."""
    manifest_path = os.path.splitext(out_path)[0] + ".frames.txt"
    lines = [os.path.join(frame_dir, f"{fid}_depth.png") for fid in clouds]
    with open(manifest_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {len(lines)} contributing frame paths -> {manifest_path}")
    return manifest_path


def save_frame_poses(clouds, detections, final_poses, out_path):
    """Write a sidecar <capture>.poses.npz (aligned with the frames manifest)
    holding each frame's final_poses (camera-local -> reference), raw T_marker_cam,
    and K -- enough to reproject/evaluate without re-running marker detection.
    Returns the path, or None if detections/final_poses are unavailable."""
    if not detections or not final_poses:
        return None
    poses_path = os.path.splitext(out_path)[0] + ".poses.npz"
    frame_ids = list(clouds)
    np.savez(
        poses_path,
        frame_ids=np.array(frame_ids),
        final_poses=np.stack([np.asarray(final_poses[fid]) for fid in frame_ids]),
        T_marker_cam=np.stack([np.asarray(detections[fid]["T_marker_cam"]) for fid in frame_ids]),
        K=np.stack([np.asarray(detections[fid]["K"]) for fid in frame_ids]))
    print(f"saved {len(frame_ids)} frame poses -> {poses_path}")
    return poses_path


def save_reconstruction(frame_dir, clouds, detections=None, final_poses=None,
                        visualize_isolation=False):
    """Isolate the target tree, filter the merged cloud, and write it plus
    sidecars (frames manifest, poses). Before-state snapshots are also written:
    the full multi-tree scene as *_scene.ply when isolation drops anything, and
    the isolated-but-unfiltered merge as *_prefilter.ply when a filter runs. The
    visibility filter takes precedence over the support filter when both are
    enabled and needs the per-frame depth maps + poses. Returns the main .ply path."""
    out_path = output_path_for(frame_dir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stem = os.path.splitext(out_path)[0]
    scene_path = stem + "_scene.ply"
    prefilter_path = stem + "_prefilter.ply"

    scene = merge_clouds(clouds)                     # full scene: every tree, before isolation
    if TREE_ISOLATION_ENABLED and final_poses and clouds:
        clouds = isolate_target_tree(clouds, final_poses, visualize=visualize_isolation)
    prefilter = merge_clouds(clouds)                 # target tree only, before filtering
    isolated = len(prefilter.points) < len(scene.points)

    merged = prefilter
    filtered = False

    if VIS_ENABLED and detections and final_poses and len(clouds) >= 2:
        merged, _ = filter_by_visibility(clouds, detections, final_poses)
        filtered = True
        superseded = " (support filter superseded)" if SUPPORT_ENABLED else ""
        print(f"visibility filter (tol={VIS_TOL_M * 1000:.0f}mm, drop if "
              f"<{VIS_MIN_CONFIRMATIONS} confirm & >={VIS_MIN_CONTRADICTIONS} see-through): "
              f"kept {len(merged.points)}/{len(prefilter.points)} points{superseded}")
    elif SUPPORT_ENABLED and len(clouds) >= SUPPORT_MIN_FRAMES:
        merged, _ = filter_by_frame_support(clouds)
        filtered = True
        print(f"support filter (r={SUPPORT_RADIUS * 1000:.0f}mm, "
              f">={SUPPORT_MIN_FRAMES} frames): kept "
              f"{len(merged.points)}/{len(prefilter.points)} points")
    elif VIS_ENABLED:
        print(f"visibility filter skipped: needs depth maps/poses and >= 2 frames "
              f"(have {len(clouds)} frame(s))")
    elif SUPPORT_ENABLED:
        print(f"support filter skipped: only {len(clouds)} frame(s), "
              f"needs >= {SUPPORT_MIN_FRAMES}")

    if isolated:
        o3d.io.write_point_cloud(scene_path, scene)
        print(f"before removing other trees ({len(scene.points)} points) -> {scene_path}")
    elif os.path.exists(scene_path):      # drop a stale sidecar from a previous run
        os.remove(scene_path)

    if filtered:
        o3d.io.write_point_cloud(prefilter_path, prefilter)
        print(f"before filtering -> {prefilter_path}")
    elif os.path.exists(prefilter_path):  # drop a stale sidecar from a previous run
        os.remove(prefilter_path)

    o3d.io.write_point_cloud(out_path, merged)
    print(f"saved reconstruction ({len(merged.points)} points) -> {out_path}")
    save_frame_manifest(frame_dir, clouds, out_path)
    save_frame_poses(clouds, detections, final_poses, out_path)
    return out_path


def reconstruct_capture(frame_dir, visualize_sam3=False, visualize_reconstruction=False,
                        visualize_isolation=False):
    """Reconstruct one capture (a .../marker_captures/depth directory): detect the
    marker in every frame, register all frames into the reference frame, save the
    merged per-frame-colored tree, and optionally show it. Returns the .ply path.
    The three visualize_* toggles each open a blocking view -- the SAM 3 overlays,
    the tree-isolation views, and the final merged Open3D window."""
    frame_ids = discover_frame_ids(frame_dir)
    print(f"found {len(frame_ids)} frames in {frame_dir}")

    detections = detect_all_markers(frame_dir, frame_ids, visualize=visualize_sam3)
    ref_id, clouds, final_poses = register_to_reference(detections)
    print(f"registered {len(clouds)}/{len(frame_ids)} frames to frame {ref_id}")

    out_path = save_reconstruction(frame_dir, clouds, detections, final_poses,
                                   visualize_isolation=visualize_isolation)

    if visualize_reconstruction:
        geoms = [vz.coord_frame(0.15)] + list(clouds.values())
        o3d.visualization.draw_geometries(
            geoms, window_name=f"tree reconstruction (reference: frame {ref_id})")
    return out_path


def main():
    reconstruct_capture(config.CFG.frame_dir,
                        visualize_sam3=config.CFG.visualize_sam3,
                        visualize_reconstruction=config.CFG.visualize_reconstruction,
                        visualize_isolation=config.CFG.visualize_isolation)


if __name__ == "__main__":
    main()
