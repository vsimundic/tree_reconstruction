"""
marker_detection_sam3.py
=========================
Detects the panel marker in a depth+intensity frame using SAM 3 for 2-D
region proposals and 3-D model registration for the final result.

Requires the host SAM 3 service (see sam3_client.py). Run:
    python marker_detection_sam3.py
Override the service address with the SAM3_URL env var if needed.
"""

import os
import re
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

import config
import depth_regions as dr
import marker_registration as reg
import sam3_client
import viz_utils as vz


# ===========================================================================
# PATHS  (from cfg/default.yaml -- see config.py)
# ===========================================================================
FRAME_DIR = config.CFG.frame_dir
GROUND_TRUTH_YAML_NAME = "marker_manual_pos.yaml"


# ===========================================================================
# SAM 3 CONFIG  (from cfg/default.yaml)
# ===========================================================================

SAM3_MULTIMASK             = config.CFG.sam3.multimask
SAM3_MIN_PROMPT_COVER      = config.CFG.sam3.min_prompt_cover       # refined mask must cover >= this fraction of the region's own mask
SAM3_MAX_AREA_RATIO        = config.CFG.sam3.max_area_ratio         # reject refined masks bigger than this x the raw region's area
SAM3_MAX_REGIONS_PER_FRAME = config.CFG.sam3.max_regions_per_frame  # cap on SAM3 calls per frame
SAM3_SAVE_STEP_IMAGES      = getattr(config.CFG.sam3, "save_step_images", False)
SAM3_STEP_IMAGES_DIR       = getattr(
    config.CFG.sam3, "step_images_dir", "/workspace/data/marker_detection_steps")

# ===========================================================================

# ground-truth yaml is cached per path so repeated frame lookups don't
# re-read the file
_GROUND_TRUTH_CACHE = {}


def _pick_best_mask(masks, scores, region_mask):
    """Best SAM 3 mask matching `region_mask` (score + IoU), gated by
    cover/area ratio so a wandering or exploded mask is rejected. None if
    nothing qualifies."""
    region_bool = region_mask > 0
    region_area = int(region_bool.sum())
    if region_area == 0:
        return None
    best, best_key = None, -1.0
    for mask, score in zip(masks, np.asarray(scores).ravel()):
        mb = mask > 0
        inter = int(np.logical_and(mb, region_bool).sum())
        cover = inter / region_area
        area_ratio = int(mb.sum()) / region_area
        if cover < SAM3_MIN_PROMPT_COVER or area_ratio > SAM3_MAX_AREA_RATIO:
            continue
        union = int(np.logical_or(mb, region_bool).sum())
        iou = inter / max(union, 1)
        key = float(score) * (0.5 + 0.5 * iou)  # score, tie-broken by IoU
        if key > best_key:
            best_key, best = key, mb
    return best


def region_masks(labels, nlab, shape):
    """Raw per-component masks (not filled contours, so depth-dropout holes
    survive) from dr.segment_regions -- used only for centroids and as SAM 3's
    acceptance bar, never sent as the prompt itself."""
    return dr.component_masks(labels, nlab, shape)


def region_centroid(mask):
    """(x, y) centroid via pixel-coordinate mean (robust to holey/non-convex masks)."""
    ys, xs = np.nonzero(mask)
    return (int(round(xs.mean())), int(round(ys.mean())))


def refine_region_with_sam3(image, region_mask, trace=None):
    """Prompt SAM 3 with only a region's centroid point (no mask), then score
    the result as a rectangle hypothesis. Acceptance is checked against the
    FULL region_mask, so SAM 3 must actually recover the true shape, not just
    touch the point. Returns a hypothesis dict (source='sam3_centroid') or
    None; raises sam3_client.Sam3ServerError on failure (no fallback). If
    `trace` is a list, appends {'centroid', 'picked', 'metrics'} per call."""
    centroid = region_centroid(region_mask)
    out = sam3_client.segment(image, points=[list(centroid)], point_labels=[1],
                              multimask=SAM3_MULTIMASK)

    picked = _pick_best_mask(out["masks"], out["scores"], region_mask)
    metrics = None
    if picked is not None:
        cnts, _ = cv2.findContours(picked.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            metrics = dr.score_rectangle_shape(c)
            if metrics is not None:
                metrics["source"] = "sam3_centroid"
                metrics["refined_mask"] = picked

    if trace is not None:
        trace.append(dict(centroid=centroid, picked=picked, metrics=metrics))
    return metrics


# --------------------------------------------------------------------------
# ground truth -- EVALUATION ONLY (see module docstring); never fed to SAM3
# --------------------------------------------------------------------------

def get_ground_truth_point(frame_dir, frame_id):
    """EVALUATION ONLY: manual ground-truth click for `frame_id`, from the
    yaml beside `frame_dir`. Never used in detection. None if missing."""
    yaml_path = os.path.join(
        os.path.dirname(frame_dir.rstrip("/")), GROUND_TRUTH_YAML_NAME)
    if yaml_path not in _GROUND_TRUTH_CACHE:
        try:
            with open(yaml_path) as f:
                _GROUND_TRUTH_CACHE[yaml_path] = yaml.safe_load(f) or {}
        except FileNotFoundError:
            _GROUND_TRUTH_CACHE[yaml_path] = {}
    entry = _GROUND_TRUTH_CACHE[yaml_path].get(f"frame{frame_id}")
    return (int(entry[0]), int(entry[1])) if entry is not None else None


def eval_top_result(results, frame_dir, frame_id):
    """Pixel distance from the top result to ground truth,
    plus its fitness/relief_fitness/accepted verdict. None if no ground
    truth or no results."""
    gt = get_ground_truth_point(frame_dir, frame_id)
    if gt is None or not results:
        return None
    top = results[0]
    bbox = top["cand"]["bbox"]
    cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    dist = float(np.hypot(cx - gt[0], cy - gt[1]))
    return dict(dist_px=dist, fitness=top["fitness"], relief_fitness=top["relief_fitness"],
               accepted=top["accepted"], source=top["cand"].get("source", "depth"))


# --------------------------------------------------------------------------
# small drawing helpers for the per-step panels
# --------------------------------------------------------------------------

def _region_colors(n, seed=0):
    rng = np.random.default_rng(seed)
    return [tuple(int(v) for v in rng.integers(60, 255, size=3)) for _ in range(n)]


def _color_overlay(intensity8, masks, colors, alpha=0.45):
    """Each mask filled in its own color over intensity8, semi-transparent."""
    vis = cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    for mask, color in zip(masks, colors):
        overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)


def _draw_centroid_prompts(intensity8, centroids, colors):
    """Each region's centroid prompt as a small dot + crosshair."""
    vis = cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)
    for (cx, cy), color in zip(centroids, colors):
        cv2.circle(vis, (cx, cy), 4, color, -1)
        cv2.drawMarker(vis, (cx, cy), color, cv2.MARKER_CROSS, 14, 1)
    return vis


def _capture_output_parts(frame_dir):
    """Return stable path components for .../<side>/<capture>/marker_captures/depth."""
    if not frame_dir:
        return ("unknown_capture",)
    path = Path(frame_dir.rstrip("/"))
    try:
        capture = path.parents[1].name
        side = path.parents[2].name
    except IndexError:
        return (path.name or "unknown_capture",)
    return tuple(part for part in (side, capture) if part)


def save_step_images(panels, output_root, frame_dir=None, frame_id=None):
    """Save every pipeline panel independently; return the created directory."""
    frame_text = str(frame_id) if frame_id is not None else "unknown"
    if frame_text.isdigit():
        frame_text = f"{int(frame_text):04d}"
    else:
        frame_text = re.sub(r"[^a-zA-Z0-9_-]+", "_", frame_text).strip("_") or "unknown"
    frame_name = f"frame_{frame_text}"
    output_dir = Path(output_root).expanduser().joinpath(
        *_capture_output_parts(frame_dir), frame_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step, (title, image) in enumerate(panels, start=1):
        # Drop the displayed numeric prefix because the filename supplies a
        # zero-padded step number.
        label = re.sub(r"^\d+\.\s*", "", title)
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        destination = output_dir / f"{step:02d}_{slug}.png"
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Failed to save marker-detection step: {destination}")

    print(f"[sam3] saved {len(panels)} step images to {output_dir}")
    return output_dir


def run_pipeline(depth_m, intensity, intensity8, K):
    """2-D segmentation -> SAM3 centroid prompts -> rescoring -> NMS ->
    reg.verify_candidates' 3-D fit, capturing one panel per step (numbered
    1-13 to match the montage). Returns (panels, hyps, results). Raises
    sam3_client.Sam3ServerError if SAM 3 fails -- no depth-only fallback.

    1-2 input | 3-4 dr.segment_regions | 5-6 raw/merged labels |
    7 all contours | 8 SAM3 centroid prompts | 9 raw SAM3 output |
    10 rescored | 11 final hyps (NMS) | 12 registration verdicts |
    13 best-fit reprojection
    """
    info = sam3_client.health()
    print(f"[sam3] connected: device={info.get('device')} "
         f"mask_input_hw={info.get('mask_input_hw')}")

    panels = [
        ("1. intensity (brightened)", intensity8),
        ("2. depth", vz.colorize(depth_m, valid=depth_m > 0)),
    ]

    image = sam3_client.to_uint8_rgb(intensity8)
    variants, proposals = dr.sweep_region_proposals(
        depth_m, intensity, max_regions=SAM3_MAX_REGIONS_PER_FRAME)
    # Keep one representative variant in the detailed debug panels.  Panel 8
    # below shows the actual prompts pooled from every variant.
    primary = variants[0]
    depth_tol, inten_tol, use_dropout_cut = primary["setting"]
    setting_text = (f"d={depth_tol:g}m i={inten_tol} "
                    f"dropout={'on' if use_dropout_cut else 'off'}")
    panels.append((f"3. boundaries ({setting_text})", primary["disc"]))
    panels.append(("4. continuous regions (valid & !discontinuity)",
                   primary["region_mask"]))
    panels.append((f"5. connected components ({primary['n_raw']})",
                   vz.color_labels(primary["labels_raw"])))
    panels.append((f"6. merged components ({primary['nlab']})",
                   vz.color_labels(primary["labels"])))

    _, contours = dr.find_rectangles(primary["labels"], primary["nlab"], depth_m.shape)
    panels.append((f"7. region contours, all ({len(contours)})",
                   dr.draw_contours(intensity8, contours)))

    masks = [proposal["mask"] for proposal in proposals]
    colors = _region_colors(len(masks))
    centroids = [region_centroid(mk) for mk in masks]
    panels.append((f"8. SAM3 prompts -- {len(masks)} pooled from "
                   f"{len(variants)} segmentation variants",
                   _draw_centroid_prompts(intensity8, centroids, colors)))

    trace = []
    refined_hyps = []
    for proposal in proposals:
        rm = proposal["mask"]
        refined = refine_region_with_sam3(image, rm, trace=trace)
        if refined is not None:
            refined["setting"] = proposal["setting"]
            refined_hyps.append(refined)

    raw_masks = [t["picked"] for t in trace if t["picked"] is not None]
    raw_colors = [c for t, c in zip(trace, colors) if t["picked"] is not None]
    panels.append((f"9. SAM3 raw output ({len(raw_masks)}/{len(masks)} pass cover/area gate)",
                   _color_overlay(intensity8, raw_masks, raw_colors)))

    rescored_panel = cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)
    for h in refined_hyps:
        cv2.polylines(rescored_panel, [h["approx"]], True, (40, 220, 40), 2)
        x, y = h["bbox"][0], h["bbox"][1]
        cv2.putText(rescored_panel, f"{h['score']:.2f}", (int(x), max(int(y) - 4, 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 220, 40), 1, cv2.LINE_AA)
    panels.append((f"10. SAM3-refined, after rescoring ({len(refined_hyps)}/{len(raw_masks)})",
                   rescored_panel))

    refined_hyps.sort(key=lambda d: -d["score"])
    kept = dr.nms_by_center(refined_hyps, dr.NMS_DIST_PX, dr.TOP_K)
    panels.append((f"11. final hypotheses, after NMS ({len(kept)})",
                   dr.draw_candidates(intensity8, kept) if kept else
                   cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)))

    results = reg.verify_candidates(kept, depth_m, K)
    panels.append((f"12. registration verdicts ({len(results)})",
                   reg.draw_verified(intensity8, results, K)))
    panels.append(("13. best-fit model reprojection",
                   reg.project_model_overlay(intensity8, results[0], K) if results else
                   cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)))

    return panels, kept, results


def detect_marker(depth_m, intensity, intensity8, K, visualize=False,
                  frame_dir=None, frame_id=None):
    """Detect the marker pose in an already-loaded frame (see dr.load_frame
    for depth_m/intensity/intensity8/K). Returns the accepted 4x4
    model-to-camera transform, or None. Raises sam3_client.Sam3ServerError
    on SAM 3 failure -- no fallback. If `visualize`, also prints results and
    shows the montage + open3d scene; `frame_dir`/`frame_id` are only for
    window titles and EVALUATION-ONLY ground-truth reporting."""
    panels, _, results = run_pipeline(depth_m, intensity, intensity8, K)

    if SAM3_SAVE_STEP_IMAGES:
        save_step_images(
            panels, SAM3_STEP_IMAGES_DIR, frame_dir=frame_dir, frame_id=frame_id)

    if visualize:
        reg.print_results(frame_id, results)
        if frame_dir is not None:
            eva = eval_top_result(results, frame_dir, frame_id)  # EVAL ONLY
            if eva is not None:
                print(f"   [eval] top result {eva['dist_px']:.1f}px from ground truth "
                     f"fitness={eva['fitness']:.3f} relief={eva['relief_fitness']:.3f} "
                     f"accepted={eva['accepted']}")

        grid = vz.montage(panels, cols=3, tile_w=380)
        vz.show_scrollable(f"SAM3 pipeline steps -- frame {frame_id}", grid)

        geoms = reg.build_scene_geometries(depth_m, intensity, K, results)
        o3d.visualization.draw_geometries(
            geoms, window_name=f"SAM3 + 3-D fit (green=accepted) -- frame {frame_id}")
        cv2.destroyAllWindows()

    if results and results[0]["accepted"]:
        return results[0]["transform"]
    return None


if __name__ == "__main__":
    for frame_id in range(config.CFG.frames.start, config.CFG.frames.end):
        intensity, intensity8, depth_m, K = dr.load_frame(FRAME_DIR, frame_id)
        pose = detect_marker(depth_m, intensity, intensity8, K, visualize=True,
                             frame_dir=FRAME_DIR, frame_id=frame_id)
        print(f"[frame {frame_id}] pose:\n{pose}")
