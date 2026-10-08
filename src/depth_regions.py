"""
depth_regions.py
================
Stage 1 of marker detection: turn a ToF depth+intensity frame into scored
rectangle-like region hypotheses. Extracted verbatim (logic unchanged) from
the original localize_rectangle.py -- only the subset marker_detection_sam3.py
actually uses. The original full script is archived under deprecated/.

The panel is bumpy (a cube and an L-shape stand on it), so we do NOT segment
by smoothness -- that would split the panel apart. Instead we segment surfaces
by DEPTH CONTINUITY: flood-connect neighbouring pixels unless a depth step (or
a dropout border) separates them. The whole panel is then one connected
component whose OUTLINE is rectangular, which we test for "rectangle-ness".

Pure 2-D: depends only on cv2/numpy/scipy/viz_utils (no open3d). The 3-D model
fit lives in marker_registration.py.
"""

import cv2
import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

import viz_utils as vz

# ===========================================================================
# CONFIG
# ===========================================================================


DEPTH_SCALE = 0.001    # raw depth units -> metres

# --- region segmentation (depth + intensity continuity) ---
DEPTH_TOL = 0.06       # depth step (m) that separates surfaces
# No single threshold works for every view: depending on pose/background the
# panel may be separated by depth, intensity, or only a dropout border.  Pool
# proposals from this small sweep and both dropout-cut states, then deduplicate
# before the (much more expensive) SAM 3 calls.
SWEEP = [
    (DEPTH_TOL, None),
    (DEPTH_TOL, 30),
    (DEPTH_TOL, 80),
    (0.08, None),
    (0.03, 50),
]
SWEEP_DROPOUT_CUTS = (True, False)
# The ToF sensor drops out (zero depth) right at object edges, so the real surface
# boundaries on this data are the valid<->invalid transitions, not adjacent depth
# steps. Treat that dropout border as a cut so segmentation follows true edges.
USE_DROPOUT_CUT = True
# Nearest-valid-neighbour step detection: for each valid pixel, compare its depth
# against the nearest valid pixel within NN_RADIUS pixels (across dropout gaps).
# Catches depth jumps hidden inside bands of missing depth. Set to 0 to disable.
NN_RADIUS   = 8        # max gap (px) to bridge when looking for a valid neighbour
DISC_DILATE = 1        # thicken the cut lines so surfaces fully separate (px radius)
MORPH_KSIZE = 5        # open kernel to despeckle the region mask

# --- merge fragmented segments (reconnect panel pieces split by the cube/L) ---
MERGE_ENABLE        = True    # reconnect marker sub-regions split by the cube/L
MERGE_GAP_PX        = 12      # segments within this pixel gap are merge candidates
MERGE_DEPTH_TOL     = 0.12    # merge if mean depth differs by < this (m)
MERGE_USE_INTENSITY = False   # depth alone is enough to reconnect marker parts
MERGE_INTENSITY_TOL = 600.0   # (unused when MERGE_USE_INTENSITY = False)

# --- rectangle test (loose, for recall -- we keep multiple hypotheses) ---
APPROX_EPS         = 0.04   # approxPolyDP epsilon as fraction of perimeter
MIN_AREA_PX        = 150    # reject tiny regions (small distant markers can be ~200 px)
MAX_AREA_FRAC      = 0.25   # reject huge regions (fraction of image)
MAX_ASPECT         = 4.5    # at slant angle θ the panel projects to aspect ~1/cos(θ)
MIN_RECTANGULARITY = 0.50   # contourArea / minAreaRect-area (1.0 = perfect rect)
MAX_CORNERS        = 8      # cube/L notch the silhouette into 6-8 corners on some views

# --- multiple hypotheses ---
TOP_K       = 20     # how many marker hypotheses to return per frame
NMS_DIST_PX = 35     # suppress duplicate hypotheses whose centres are closer than this

# ===========================================================================


def load_camera(yaml_path):
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    K = np.array(d['k'], float).reshape(3, 3)
    return K, int(d['width']), int(d['height'])


def load_frame(frame_dir, frame_id):
    """Load one frame and convert it to the arrays used by the pipeline."""
    intensity = cv2.imread(f'{frame_dir}/{frame_id}_intensity.png',
                           cv2.IMREAD_UNCHANGED)
    depth_raw = cv2.imread(f'{frame_dir}/{frame_id}_depth.png',
                           cv2.IMREAD_UNCHANGED)
    K, _, _ = load_camera(f'{frame_dir}/{frame_id}_camera_params.yaml')

    if intensity.ndim == 3:
        intensity = cv2.cvtColor(intensity, cv2.COLOR_BGR2GRAY)

    intensity = intensity.astype(np.float32)
    depth_m = depth_raw.astype(np.float32) * DEPTH_SCALE
    intensity8 = vz.brighten(intensity, gamma=0.5)
    return intensity, intensity8, depth_m, K


# --------------------------------------------------------------------------
# depth-continuity segmentation
# --------------------------------------------------------------------------

def fill_depth_nn(depth_m, radius):
    """Fill invalid (zero) pixels with their nearest valid pixel's depth, if
    within `radius` px. Valid pixels unchanged."""
    invalid = depth_m == 0
    if not invalid.any():
        return depth_m.copy()
    dist, idx = distance_transform_edt(invalid, return_indices=True)
    filled = depth_m.copy()
    to_fill = invalid & (dist <= radius)
    filled[to_fill] = depth_m[idx[0][to_fill], idx[1][to_fill]]
    return filled


def _neighbour_step(field, valid, tol):
    """Pixels on a step > tol from a valid 4-neighbour of `field`."""
    disc = np.zeros(field.shape, np.uint8)
    dh = np.abs(field[:, 1:] - field[:, :-1]); mh = valid[:, 1:] & valid[:, :-1] & (dh > tol)
    disc[:, 1:][mh] = 1; disc[:, :-1][mh] = 1
    dv = np.abs(field[1:, :] - field[:-1, :]); mv = valid[1:, :] & valid[:-1, :] & (dv > tol)
    disc[1:, :][mv] = 1; disc[:-1, :][mv] = 1
    return disc


def discontinuities(depth_m, intensity=None, depth_tol=0.06, inten_tol=None):
    """Binary map of surface boundaries: depth/intensity steps, dropout
    borders, and nearest-valid-neighbour depth steps across gaps -- OR'd
    together."""
    valid = depth_m > 0
    disc = _neighbour_step(depth_m, valid, depth_tol)
    if inten_tol is not None and intensity is not None:
        iv = intensity[valid]
        lo, hi = np.percentile(iv, 2), np.percentile(iv, 98)
        norm = np.clip((intensity - lo) / (hi - lo + 1e-6), 0, 1) * 255.0
        disc |= _neighbour_step(norm, valid, inten_tol)
    if USE_DROPOUT_CUT:
        valid_u8 = valid.astype(np.uint8)
        eroded = cv2.erode(valid_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        disc |= valid_u8 & (1 - eroded)
    if NN_RADIUS > 0:
        filled = fill_depth_nn(depth_m, NN_RADIUS)
        # filled pixels that were originally invalid now carry the nearest valid depth,
        # so a valid pixel on one side of a gap is compared against the filled value on
        # the other side. valid_filled ensures both sides of a comparison have depth.
        valid_filled = filled > 0
        disc |= _neighbour_step(filled, valid_filled, depth_tol)
    if DISC_DILATE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (DISC_DILATE * 2 + 1, DISC_DILATE * 2 + 1))
        disc = cv2.dilate(disc, k)
    return disc * 255


def segment_regions(depth_m, intensity=None, depth_tol=0.06, inten_tol=None,
                    use_dropout_cut=None):
    """Segment surfaces by depth (+ optional intensity) continuity, not
    smoothness. Returns (disc, region_mask, n, labels, stats)."""
    valid = depth_m > 0
    old_dropout_cut = None
    if use_dropout_cut is not None:
        old_dropout_cut = USE_DROPOUT_CUT
        globals()['USE_DROPOUT_CUT'] = use_dropout_cut
    try:
        disc = discontinuities(depth_m, intensity, depth_tol, inten_tol)
    finally:
        if old_dropout_cut is not None:
            globals()['USE_DROPOUT_CUT'] = old_dropout_cut
    region = valid & (disc == 0)
    region_mask = region.astype(np.uint8) * 255
    # open only (do NOT close -- closing would bridge back across the depth cuts)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KSIZE, MORPH_KSIZE))
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(region_mask, 8)
    return disc, region_mask, n, labels, stats


def merge_segments(labels, n, depth_m, intensity=None):
    """Union spatially-close, depth-consistent segments so panel pieces split
    by the cube/L reconnect. Returns (merged_labels, merged_count)."""
    if n <= 1:
        return labels.copy(), 0
    valid = depth_m > 0

    # per-segment mean depth / intensity and mask
    masks, mean_d, mean_i = {}, {}, {}
    for i in range(1, n):
        m = labels == i
        masks[i] = m.astype(np.uint8)
        dv = depth_m[m & valid]
        mean_d[i] = float(dv.mean()) if dv.size else 0.0
        if intensity is not None:
            iv = intensity[m]
            mean_i[i] = float(iv.mean()) if iv.size else 0.0

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # adjacency: dilate each segment, see which other labels it touches
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MERGE_GAP_PX, MERGE_GAP_PX))
    for i in range(1, n):
        neigh = np.unique(labels[cv2.dilate(masks[i], k) > 0])
        for j in neigh:
            if j <= i:                       # skip background(0), self, and dupes
                continue
            if abs(mean_d[i] - mean_d[j]) > MERGE_DEPTH_TOL:
                continue
            if (MERGE_USE_INTENSITY and intensity is not None and
                    abs(mean_i[i] - mean_i[j]) > MERGE_INTENSITY_TOL):
                continue
            union(i, j)

    # relabel merged groups to contiguous ids 1..K
    merged = np.zeros_like(labels)
    roots, nxt = {}, 1
    for i in range(1, n):
        r = find(i)
        if r not in roots:
            roots[r] = nxt
            nxt += 1
        merged[labels == i] = roots[r]
    return merged, nxt - 1


# --------------------------------------------------------------------------
# rectangle-ness scoring
# --------------------------------------------------------------------------

def _corner_angles(approx):
    """Mean absolute deviation of a polygon's interior angles from 90 degrees."""
    p = approx.reshape(-1, 2).astype(np.float64)
    n = len(p)
    devs = []
    for i in range(n):
        a, b, c = p[(i - 1) % n], p[i], p[(i + 1) % n]
        v1, v2 = a - b, c - b
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        devs.append(abs(ang - 90))
    return float(np.mean(devs))


def score_rectangle_shape(c):
    """Rectangle-ness score for one contour, or None if it fails the shape
    gates (corners, convexity, aspect, rectangularity). Same rule set for
    fresh region contours and externally recovered ones (e.g. SAM3-refined)."""
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, APPROX_EPS * peri, True)
    if not (4 <= len(approx) <= MAX_CORNERS) or not cv2.isContourConvex(
            cv2.convexHull(approx)):
        return None
    (rw, rh) = cv2.minAreaRect(c)[1]
    if rw < 1 or rh < 1:
        return None
    aspect = max(rw, rh) / min(rw, rh)
    if aspect > MAX_ASPECT:
        return None
    rectangularity = area / (rw * rh)
    if rectangularity < MIN_RECTANGULARITY:
        return None
    angle_dev = _corner_angles(approx) if len(approx) == 4 else 20.0
    quad_bonus = 1.25 if len(approx) == 4 else 1.0
    # Score is pure shape quality -- no area term -- so small, distant markers score
    # the same as large, close ones with equal shape, and range doesn't bury detections.
    score = rectangularity / aspect * quad_bonus \
        * (1.0 - min(angle_dev, 45) / 90.0)
    return dict(contour=c, approx=approx, bbox=(x, y, w, h),
               center=(x + w / 2, y + h / 2),
               area=int(area), aspect=round(aspect, 2),
               rectangularity=round(float(rectangularity), 3),
               corners=len(approx), angle_dev=round(angle_dev, 1),
               score=float(score))


def find_rectangles(labels, n, shape):
    """Score every region's boundary contour for rectangle-ness. Returns
    (candidates_sorted_best_first, all_contours)."""
    H, W = shape
    cands, all_contours = [], []
    for i in range(1, n + 1):
        comp = (labels == i).astype(np.uint8)
        region_area = int(comp.sum())
        if region_area < MIN_AREA_PX or region_area > MAX_AREA_FRAC * H * W:
            continue
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        all_contours.append(c)
        metrics = score_rectangle_shape(c)
        if metrics is not None:
            cands.append(metrics)
    cands.sort(key=lambda d: -d['score'])
    return cands, all_contours


def nms_by_center(cands, dist_px, top_k):
    """Greedy NMS by center distance: keep a candidate only if farther than
    `dist_px` from all kept ones, up to `top_k`. Earlier entries in `cands`
    win ties."""
    kept = []
    for c in cands:
        cx, cy = c['center']
        if all((cx - k['center'][0]) ** 2 + (cy - k['center'][1]) ** 2 > dist_px ** 2
               for k in kept):
            kept.append(c)
        if len(kept) >= top_k:
            break
    return kept


# --------------------------------------------------------------------------
# multi-setting region proposals
# --------------------------------------------------------------------------

def component_masks(labels, nlab, shape):
    """Size-gated component masks from a merged label image.

    The masks remain holey/non-convex on purpose: they are proposal seeds and
    acceptance references for SAM 3, not claimed marker silhouettes.
    """
    H, W = shape
    masks = []
    for i in range(1, nlab + 1):
        comp = labels == i
        area = int(comp.sum())
        if area < MIN_AREA_PX or area > MAX_AREA_FRAC * H * W:
            continue
        masks.append(comp.astype(np.uint8) * 255)
    return masks


def _mask_iou(a, b):
    aa, bb = a > 0, b > 0
    inter = int(np.logical_and(aa, bb).sum())
    if inter == 0:
        return 0.0
    return inter / int(np.logical_or(aa, bb).sum())


def sweep_region_proposals(depth_m, intensity, max_regions=None, dedupe_iou=0.85):
    """Run every ``SWEEP`` setting with and without dropout cuts.

    Returns ``(variants, proposals)``.  Each variant retains its intermediate
    segmentation products for visualisation.  Proposals are component masks
    pooled across variants, near-identical masks are removed by IoU, and the
    result is ordered largest-first to preserve the old single-setting policy.
    ``max_regions`` is applied only after pooling/deduplication.
    """
    variants = []
    pooled = []
    for depth_tol, inten_tol in SWEEP:
        for use_dropout_cut in SWEEP_DROPOUT_CUTS:
            disc, region_mask, n, labels_raw, _ = segment_regions(
                depth_m, intensity, depth_tol, inten_tol,
                use_dropout_cut=use_dropout_cut)
            if MERGE_ENABLE:
                labels, nlab = merge_segments(labels_raw, n, depth_m, intensity)
            else:
                labels, nlab = labels_raw, n - 1

            setting = (depth_tol, inten_tol, use_dropout_cut)
            sizes = np.bincount(labels.ravel(), minlength=nlab + 1)
            variants.append(dict(
                setting=setting,
                disc=disc,
                region_mask=region_mask,
                n_raw=n - 1,
                labels_raw=labels_raw,
                labels=labels,
                nlab=nlab,
            ))
            H, W = depth_m.shape
            variant_index = len(variants) - 1
            pooled.extend(dict(variant_index=variant_index, label=label,
                               setting=setting, area=int(sizes[label]))
                          for label in range(1, nlab + 1)
                          if MIN_AREA_PX <= sizes[label] <= MAX_AREA_FRAC * H * W)

    # The former single-setting path considered larger components first.  Keep
    # that behaviour, while preferring the first sweep setting when areas tie.
    pooled.sort(key=lambda p: -p["area"])
    proposals = []
    for proposal in pooled:
        variant = variants[proposal["variant_index"]]
        mask = (variant["labels"] == proposal["label"]).astype(np.uint8) * 255
        if any(_mask_iou(mask, kept["mask"]) >= dedupe_iou
               for kept in proposals):
            continue
        proposals.append(dict(mask=mask, setting=proposal["setting"],
                              area=proposal["area"]))
        if max_regions is not None and len(proposals) >= max_regions:
            break
    return variants, proposals


# --------------------------------------------------------------------------
# visualization
# --------------------------------------------------------------------------

def draw_contours(intensity8, contours, approxes=None):
    vis = cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, contours, -1, (0, 140, 255), 1)
    if approxes:
        for ap in approxes:
            cv2.polylines(vis, [ap], True, (255, 120, 0), 2)
    return vis


def draw_candidates(intensity8, cands):
    vis = cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)
    for rank, c in enumerate(cands):
        color = (40, 220, 40) if rank == 0 else (200, 160, 0)
        cv2.polylines(vis, [c['approx']], True, color, 2)
        for pt in c['approx'].reshape(-1, 2):
            cv2.circle(vis, tuple(int(v) for v in pt), 4, color, -1)
        x, y = c['approx'].reshape(-1, 2)[0]
        cv2.putText(vis, f"#{rank} r={c['rectangularity']} asp={c['aspect']} n={c['corners']}",
                    (int(x), max(int(y) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    color, 1, cv2.LINE_AA)
    return vis
