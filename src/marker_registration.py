"""
marker_registration.py
======================
Stage 2 of marker detection: verify each 2-D rectangle hypothesis by
registering the known marker mesh against the ROI's depth cloud, returning a
full 6-DoF pose. Extracted verbatim (logic unchanged) from the original
localize_rectangle_hybrid.py -- only the subset marker_detection_sam3.py uses,
plus the three ROI/lift geometry helpers it depended on from
localize_rectangle.py. The original full scripts are archived under deprecated/.

Per candidate: lift a padded bbox ROI to a 3-D point cloud, FPFH+RANSAC for a
coarse pose of the marker mesh, then point-to-plane ICP to refine it. Accept/
reject by two VIEW-DEPENDENT fitness numbers -- the matched fraction of
camera-facing model points (fitness) and of camera-facing cube/L relief points
(relief_fitness) -- plus inlier RMSE. Requiring both is what rejects imposters:
flat ground can imitate the panel but not the relief, and noisy vegetation
matches neither. Results are ranked by fitness * relief_fitness and deduped by
3-D marker centre; results[0] (if accepted) is the detection, and its transform
maps marker-model coordinates (metres) into the camera frame.
"""

import cv2
import numpy as np
import open3d as o3d

import config
import viz_utils as vz

# ===========================================================================
# CONFIG
# ===========================================================================

MODEL_PATH = config.CFG.model_path        # from cfg/default.yaml
MODEL_UNITS_TO_M = config.CFG.model_units_to_m

# --- proposal stage ---
MAX_CANDIDATES = 20     # how many 2-D hypotheses to verify with registration
ROI_PAD_PX     = 25     # pad around the candidate bbox before lifting to 3-D

# --- registration ---
VOXEL = 0.006                      # working resolution (m); relief is 0.055 m
NORMAL_RADIUS   = VOXEL * 2.5
FPFH_RADIUS     = VOXEL * 5.0
RANSAC_DIST     = VOXEL * 1.5      # max correspondence distance, coarse stage
RANSAC_ITERS    = 100_000
RANSAC_CONF     = 0.999
ICP_DIST        = VOXEL * 1.2      # max correspondence distance, refine stage
MODEL_SAMPLES   = 3000             # points sampled from the marker mesh
MIN_ROI_POINTS  = 200              # skip ROIs with fewer downsampled points
# RANSAC is stochastic and sometimes hands ICP a coarse pose it cannot recover
# from; retry a few times per ROI and keep the best pose by visible fitness
# (early exit once a pose passes acceptance).
REG_ATTEMPTS    = 3
RANDOM_SEED     = 0                # open3d global seed; reduces run-to-run variance
                                   # (RANSAC is multithreaded, so not fully repeatable)
# The camera can never see the underside of the panel, so bottom-facing model
# points (normal ~ -Z in model frame) are culled -- otherwise they cap the best
# possible fitness at ~0.6 and drown the FPFH matching in unmatchable points.
CULL_BOTTOM_NZ  = -0.3             # drop model points with normal_z below this
RELIEF_MIN_Z_M  = 0.02             # model points above this height = cube/L relief

# --- acceptance ---
# Both fitness numbers are VIEW-DEPENDENT: they count only model points whose
# normal faces the camera under the estimated pose, so self-occluded surfaces
# do not drag a correct pose down. Requiring BOTH rejects imposters: noisy
# vegetation can reach f~0.43 but not rf, flat ground reaches neither.
MIN_FITNESS        = 0.40   # matched fraction of camera-facing model points
MIN_RELIEF_FITNESS = 0.40   # matched fraction of camera-facing cube/L points
MAX_RMSE_M         = 0.008  # mean inlier residual
DEDUPE_DIST_M      = 0.15   # overlapping ROIs converge to the same marker pose;
                            # keep only the best result per 3-D location

# ===========================================================================

_MODEL_CACHE = None


# --------------------------------------------------------------------------
# ROI / lift geometry (was shared with localize_rectangle.py)
# --------------------------------------------------------------------------

def bbox_roi_mask(shape, bbox, pad):
    x, y, w, h = bbox
    H, W = shape[:2]
    x0 = max(int(x) - pad, 0)
    y0 = max(int(y) - pad, 0)
    x1 = min(int(x + w) + pad, W)
    y1 = min(int(y + h) + pad, H)
    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def lift_mask_to_points(mask, depth_m, K, intensity=None):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.where(mask > 0)
    Z = depth_m[ys, xs]
    ok = np.isfinite(Z) & (Z > 0)
    xs, ys, Z = xs[ok], ys[ok], Z[ok]
    X = (xs - cx) / fx * Z
    Y = (ys - cy) / fy * Z
    pts = np.stack([X, Y, Z], axis=1)
    if intensity is not None:
        return pts, xs, ys, intensity[ys, xs]
    return pts, xs, ys


def lift_to_3d(mask, depth_m, K, intensity=None):
    if intensity is not None:
        pts, _, _, vals = lift_mask_to_points(mask, depth_m, K, intensity)
        return pts, vals
    pts, _, _ = lift_mask_to_points(mask, depth_m, K)
    return pts


# --------------------------------------------------------------------------
# model + registration
# --------------------------------------------------------------------------

def load_model_cloud():
    """Marker mesh -> downsampled point cloud with normals + FPFH (cached)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    mesh = o3d.io.read_triangle_mesh(MODEL_PATH)
    mesh.scale(MODEL_UNITS_TO_M, center=(0, 0, 0))
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(MODEL_SAMPLES)
    keep = np.where(np.asarray(pcd.normals)[:, 2] > CULL_BOTTOM_NZ)[0]
    pcd = pcd.select_by_index(keep)
    pcd = pcd.voxel_down_sample(VOXEL)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_RADIUS, max_nn=100))

    points = np.asarray(pcd.points)
    _MODEL_CACHE = dict(pcd=pcd, fpfh=fpfh, center=points.mean(axis=0),
                        relief_idx=np.where(points[:, 2] > RELIEF_MIN_Z_M)[0])
    return _MODEL_CACHE


def roi_cloud(cand, depth_m, K):
    """Lift a padded bbox ROI to a downsampled point cloud with normals + FPFH."""
    mask = bbox_roi_mask(depth_m.shape, cand['bbox'], ROI_PAD_PX)
    pts, _, _ = lift_mask_to_points(mask, depth_m, K)
    if pts.shape[0] < MIN_ROI_POINTS:
        return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pcd = pcd.voxel_down_sample(VOXEL)
    if len(pcd.points) < MIN_ROI_POINTS:
        return None
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=30))
    # ToF camera sits at the origin; make normals face it so point-to-plane ICP
    # and FPFH signatures are consistent between frames.
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_RADIUS, max_nn=100))
    return dict(pcd=pcd, fpfh=fpfh)


def register_model_to_roi(model, roi):
    """FPFH+RANSAC coarse pose, then point-to-plane ICP. Returns (T, fitness, rmse)."""
    reg = o3d.pipelines.registration
    coarse = reg.registration_ransac_based_on_feature_matching(
        model['pcd'], roi['pcd'], model['fpfh'], roi['fpfh'],
        mutual_filter=True,
        max_correspondence_distance=RANSAC_DIST,
        estimation_method=reg.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(RANSAC_DIST),
        ],
        criteria=reg.RANSACConvergenceCriteria(RANSAC_ITERS, RANSAC_CONF),
    )
    if coarse.fitness == 0.0:
        return None, 0.0, np.inf

    icp = reg.registration_icp(
        model['pcd'], roi['pcd'], ICP_DIST, coarse.transformation,
        reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(max_iteration=60),
    )
    # icp.fitness = fraction of MODEL points with a scene correspondence within
    # ICP_DIST -- exactly the "how much of the marker did we explain" number.
    return icp.transformation, float(icp.fitness), float(icp.inlier_rmse)


def visible_fitness(model, roi, T):
    """(fitness, relief_fitness) over only camera-facing model points, so
    self-occlusion can't cap a correct pose and flat surfaces can't fake the
    relief."""
    T = np.asarray(T)
    pts = np.asarray(model['pcd'].points) @ T[:3, :3].T + T[:3, 3]
    nrm = np.asarray(model['pcd'].normals) @ T[:3, :3].T
    facing = (nrm * pts).sum(axis=1) < 0    # normal points back at the camera
    tree = o3d.geometry.KDTreeFlann(roi['pcd'])
    hit = np.array([tree.search_radius_vector_3d(p, ICP_DIST)[0] > 0
                    for p in pts])
    relief = np.zeros(len(pts), bool)
    relief[model['relief_idx']] = True
    fit = float(hit[facing].mean()) if facing.any() else 0.0
    relief_fit = (float(hit[facing & relief].mean())
                  if (facing & relief).any() else 0.0)
    return fit, relief_fit


def register_with_retries(model, roi):
    """Best pose over REG_ATTEMPTS runs, ranked by fitness * relief_fitness;
    stops early once acceptable."""
    best = None
    for _ in range(REG_ATTEMPTS):
        T, _, rmse = register_model_to_roi(model, roi)
        if T is None:
            continue
        fitness, relief_fit = visible_fitness(model, roi, T)
        if best is None or fitness * relief_fit > best[1] * best[2]:
            best = (np.asarray(T), fitness, relief_fit, rmse)
        if (fitness >= MIN_FITNESS and relief_fit >= MIN_RELIEF_FITNESS
                and rmse <= MAX_RMSE_M):
            break
    return best


def verify_candidates(cands, depth_m, K):
    """Run registration on each 2-D candidate; return results sorted by fitness."""
    model = load_model_cloud()   # before seeding: mesh sampling draws random
    o3d.utility.random.seed(RANDOM_SEED)  # numbers only on the first, uncached call
    results = []
    for rank, cand in enumerate(cands[:MAX_CANDIDATES]):
        roi = roi_cloud(cand, depth_m, K)
        if roi is None:
            continue
        best = register_with_retries(model, roi)
        if best is None:
            continue
        T, fitness, relief_fit, rmse = best
        results.append(dict(
            cand=cand,
            proposal_rank=rank,
            transform=T,
            center=T[:3, :3] @ model['center'] + T[:3, 3],
            fitness=fitness,
            relief_fitness=relief_fit,
            rmse=rmse,
            accepted=(fitness >= MIN_FITNESS and
                      relief_fit >= MIN_RELIEF_FITNESS and
                      rmse <= MAX_RMSE_M),
            roi_pcd=roi['pcd'],
        ))
    results.sort(key=lambda r: -(r['fitness'] * r['relief_fitness']))
    # overlapping ROIs find the same marker; keep the best pose per 3-D location
    deduped = []
    for r in results:
        if all(np.linalg.norm(r['center'] - k['center']) > DEDUPE_DIST_M
               for k in deduped):
            deduped.append(r)
    return deduped


def print_results(frame_id, results):
    print(f"[frame {frame_id}] {len(results)} registered candidates")
    for i, r in enumerate(results):
        c = r['center']
        flag = 'ACCEPT' if r['accepted'] else 'reject'
        print(f"   #{i} [{flag}] fitness={r['fitness']:.3f} "
              f"relief={r['relief_fitness']:.3f} rmse={r['rmse'] * 1000:.1f}mm "
              f"proposal_rank={r['proposal_rank']} bbox={r['cand']['bbox']} "
              f"center=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})m")
    if results and results[0]['accepted']:
        np.set_printoptions(precision=4, suppress=True)
        print("   best pose (model -> camera):")
        print(np.array2string(results[0]['transform'], prefix="   "))


# --------------------------------------------------------------------------
# visualization
# --------------------------------------------------------------------------

def draw_verified(intensity8, results, K):
    """Candidate quads colored by verdict: green accepted, red rejected."""
    vis = cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)
    for i, r in enumerate(results):
        color = (40, 220, 40) if r['accepted'] else (60, 60, 230)
        cv2.polylines(vis, [r['cand']['approx']], True, color, 2)
        x, y = r['cand']['bbox'][0], r['cand']['bbox'][1]
        cv2.putText(vis, f"#{i} f={r['fitness']:.2f} rf={r['relief_fitness']:.2f}",
                    (int(x), max(int(y) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    color, 1, cv2.LINE_AA)
    return vis


def project_model_overlay(intensity8, result, K):
    """Project the registered model points into the image over the intensity."""
    vis = cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)
    model = load_model_cloud()
    T = result['transform']
    pts = np.asarray(model['pcd'].points) @ T[:3, :3].T + T[:3, 3]
    Z = pts[:, 2]
    ok = Z > 1e-6
    u = (K[0, 0] * pts[ok, 0] / Z[ok] + K[0, 2]).round().astype(int)
    v = (K[1, 1] * pts[ok, 1] / Z[ok] + K[1, 2]).round().astype(int)
    H, W = intensity8.shape[:2]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    vis[v[inb], u[inb]] = (40, 220, 40)
    return vis


def build_scene_geometries(depth_m, intensity, K, results):
    all_pts, all_i = lift_to_3d((depth_m > 0).astype(np.uint8) * 255,
                                depth_m, K, intensity)
    scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_pts))
    scene.colors = o3d.utility.Vector3dVector(vz.color_by_value(all_i) * 0.6)
    geoms = [scene, vz.coord_frame(0.1)]

    model = load_model_cloud()
    for i, r in enumerate(results[:3]):
        aligned = o3d.geometry.PointCloud(model['pcd'])
        aligned.transform(r['transform'])
        aligned.paint_uniform_color((0.15, 0.85, 0.15) if r['accepted']
                                    else (0.85, 0.2, 0.2))
        geoms.append(aligned)
        if i == 0:
            geoms.append(vz.aabb(aligned, (1, 1, 0)))
    return geoms
