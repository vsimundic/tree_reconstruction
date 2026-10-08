#!/usr/bin/env python3
"""Evaluate rigid and non-rigid final clouds by projecting them into camera views.

Only the two final point clouds are scored.  For each camera, points contributed
by that same frame are excluded by default, so a cloud cannot score well merely
by projecting a frame's original points back into their source pixels.

The observed silhouette is either a supplied binary mask or, by default, the
valid-depth footprint in the raw depth image.  All settings are variables below;
there is intentionally no command-line parser.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

TREE_RECONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
if str(TREE_RECONSTRUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(TREE_RECONSTRUCTION_DIR))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import viz_utils as vz


# ---------------------------------------------------------------------------
# Configuration -- edit these variables, then run this file without arguments.
# ---------------------------------------------------------------------------
RIGID_PLY = Path(
    "data/pears/pears_reconstructed/B/tree_0_V_0000/"
    "tree_0_V_0000_merged_scene.ply"
)
NONRIGID_PLY: Path | None = None
POSES_PATH: Path | None = None
FRAMES_MANIFEST_PATH: Path | None = None
OUTPUT_DIR: Path | None = None

# Set this to a directory containing <frame_id>_mask.png files for true tree
# silhouette IoU.  With None, the observed mask is depth > 0, which evaluates
# fitting to all measured surfaces in *_merged_scene.ply.
MASK_DIR: Path | None = None
MASK_FILENAME_TEMPLATE = "{frame_id}_mask.png"

# Excluding a frame's own colour block avoids the trivial self-projection match.
EXCLUDE_SOURCE_FRAME_POINTS = True
DEPTH_SCALE_M = 0.001
MIN_PROJECTED_DEPTH_M = 0.05
RENDER_POINT_RADIUS_PX = 1
SILHOUETTE_TOLERANCE_PX = 3
DEPTH_TOLERANCES_M = (0.01, 0.02, 0.04)
DEPTH_ERROR_DISPLAY_MAX_M = 0.05
MAX_MONTAGE_FRAMES = 20


def _base_stem(rigid_path: Path) -> str:
    stem = rigid_path.stem
    return stem.removesuffix("_scene") if stem.endswith("_merged_scene") else stem


def _default_poses_path(rigid_path: Path) -> Path:
    return rigid_path.with_name(f"{_base_stem(rigid_path)}.poses.npz")


def _default_manifest_path(rigid_path: Path) -> Path:
    return rigid_path.with_name(f"{_base_stem(rigid_path)}.frames.txt")


def _default_nonrigid_path(rigid_path: Path) -> Path:
    return (
        rigid_path.parent
        / "non_rigid_comparison"
        / f"{rigid_path.stem}_nonrigid.ply"
    )


def _default_output_dir(rigid_path: Path) -> Path:
    return rigid_path.parent / "projection_fit_comparison" / rigid_path.stem


def _read_metadata(
    poses_path: Path, manifest_path: Path
) -> tuple[list[str], np.ndarray, np.ndarray, list[Path]]:
    if not poses_path.is_file():
        raise FileNotFoundError(f"pose sidecar not found: {poses_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"frame manifest not found: {manifest_path}")

    with np.load(poses_path, allow_pickle=False) as data:
        required = {"frame_ids", "final_poses", "K"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"pose sidecar is missing arrays: {sorted(missing)}")
        frame_ids = [str(value) for value in data["frame_ids"]]
        final_poses = np.asarray(data["final_poses"], dtype=np.float64)
        intrinsics = np.asarray(data["K"], dtype=np.float64)

    frame_paths = [
        Path(line.strip())
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_pose_shape = (len(frame_ids), 4, 4)
    expected_k_shape = (len(frame_ids), 3, 3)
    if final_poses.shape != expected_pose_shape:
        raise ValueError(
            f"final_poses has shape {final_poses.shape}, expected {expected_pose_shape}"
        )
    if intrinsics.shape != expected_k_shape:
        raise ValueError(
            f"K has shape {intrinsics.shape}, expected {expected_k_shape}"
        )
    if len(frame_paths) != len(frame_ids):
        raise ValueError(
            f"manifest has {len(frame_paths)} frames but poses has {len(frame_ids)}"
        )

    manifest_ids = [path.stem.removesuffix("_depth") for path in frame_paths]
    if manifest_ids != frame_ids:
        raise ValueError(
            "frame manifest order does not match frame_ids in the pose sidecar"
        )
    missing_frames = [str(path) for path in frame_paths if not path.is_file()]
    if missing_frames:
        raise FileNotFoundError(
            "depth frames listed by the manifest are unavailable: "
            + ", ".join(missing_frames[:3])
        )
    return frame_ids, final_poses, intrinsics, frame_paths


def _read_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"point cloud not found: {path}")
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    colours = np.asarray(cloud.colors, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(f"point cloud is empty: {path}")
    if colours.shape != points.shape:
        raise ValueError(
            f"point cloud must retain per-frame colours for evaluation: {path}"
        )
    rgb = np.rint(np.clip(colours, 0.0, 1.0) * 255.0).astype(np.uint8)
    return points, rgb


def _frame_ranges(rgb: np.ndarray, frame_ids: list[str]) -> list[tuple[int, int]]:
    starts = np.flatnonzero(
        np.r_[True, np.any(rgb[1:] != rgb[:-1], axis=1)]
    )
    ends = np.r_[starts[1:], len(rgb)]
    if len(starts) != len(frame_ids):
        raise ValueError(
            "could not recover one contiguous colour block per frame: "
            f"found {len(starts)} blocks for {len(frame_ids)} frames"
        )
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _load_cloud_pair(
    rigid_path: Path, nonrigid_path: Path, frame_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    rigid_points, rigid_rgb = _read_cloud(rigid_path)
    nonrigid_points, nonrigid_rgb = _read_cloud(nonrigid_path)
    if rigid_points.shape != nonrigid_points.shape:
        raise ValueError(
            "rigid and non-rigid clouds must contain the same ordered points: "
            f"{len(rigid_points)} != {len(nonrigid_points)}"
        )
    if not np.array_equal(rigid_rgb, nonrigid_rgb):
        raise ValueError(
            "rigid and non-rigid clouds do not have matching per-frame colours"
        )
    ranges = _frame_ranges(rigid_rgb, frame_ids)
    return rigid_points, nonrigid_points, ranges


def _read_frame(depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"could not read depth image: {depth_path}")
    depth_m = depth_raw.astype(np.float32) * DEPTH_SCALE_M

    intensity_path = depth_path.with_name(
        depth_path.name.replace("_depth.png", "_intensity.png")
    )
    intensity = cv2.imread(str(intensity_path), cv2.IMREAD_UNCHANGED)
    if intensity is None:
        raise FileNotFoundError(f"could not read intensity image: {intensity_path}")
    if intensity.ndim == 3:
        intensity = cv2.cvtColor(intensity, cv2.COLOR_BGR2GRAY)
    intensity8 = vz.brighten(intensity.astype(np.float32), gamma=0.5)
    return depth_m, intensity8


def _read_observed_mask(
    frame_id: str, depth_m: np.ndarray, mask_dir: Path | None
) -> tuple[np.ndarray, str]:
    if mask_dir is None:
        return depth_m > 0, "valid_depth"

    mask_path = mask_dir / MASK_FILENAME_TEMPLATE.format(frame_id=frame_id)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"tree mask not found: {mask_path}")
    if mask.shape != depth_m.shape:
        raise ValueError(
            f"mask {mask_path} has shape {mask.shape}, expected {depth_m.shape}"
        )
    return mask > 0, str(mask_path)


def _inverse_rigid(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def _render_depth(
    points_ref: np.ndarray,
    T_cam_ref: np.ndarray,
    K: np.ndarray,
    image_shape: tuple[int, int],
    excluded_range: tuple[int, int] | None,
) -> np.ndarray:
    height, width = image_shape
    sentinel = np.float32(1e6)
    rendered = np.full((height, width), sentinel, dtype=np.float32)
    T_ref_cam = _inverse_rigid(T_cam_ref)
    rotation = T_ref_cam[:3, :3]
    translation = T_ref_cam[:3, 3]

    if excluded_range is None:
        slices = (slice(None),)
    else:
        start, end = excluded_range
        slices = (slice(0, start), slice(end, len(points_ref)))

    flat = rendered.ravel()
    for point_slice in slices:
        subset = points_ref[point_slice]
        if len(subset) == 0:
            continue
        camera = subset @ rotation.T + translation
        z = camera[:, 2]
        in_front = z > MIN_PROJECTED_DEPTH_M
        camera = camera[in_front]
        z = z[in_front]
        if len(z) == 0:
            continue

        u = np.rint(camera[:, 0] / z * K[0, 0] + K[0, 2]).astype(np.int64)
        v = np.rint(camera[:, 1] / z * K[1, 1] + K[1, 2]).astype(np.int64)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indices = v[inside] * width + u[inside]
        np.minimum.at(flat, indices, z[inside].astype(np.float32))

    if RENDER_POINT_RADIUS_PX > 0:
        radius = RENDER_POINT_RADIUS_PX
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        rendered = cv2.erode(
            rendered,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=float(sentinel),
        )

    valid = rendered < sentinel * 0.5
    rendered[~valid] = np.nan
    return rendered


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _f1(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall):
        return float("nan")
    return _ratio(2.0 * precision * recall, precision + recall)


def _evaluate_projection(
    predicted_depth_m: np.ndarray,
    observed_depth_m: np.ndarray,
    observed_mask: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray]:
    predicted = np.isfinite(predicted_depth_m)
    observed = observed_mask.astype(bool)

    intersection = predicted & observed
    union = predicted | observed
    predicted_count = int(predicted.sum())
    observed_count = int(observed.sum())
    intersection_count = int(intersection.sum())
    union_count = int(union.sum())

    precision = _ratio(intersection_count, predicted_count)
    recall = _ratio(intersection_count, observed_count)

    radius = SILHOUETTE_TOLERANCE_PX
    if radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        dilated_predicted = cv2.dilate(predicted.astype(np.uint8), kernel) > 0
        dilated_observed = cv2.dilate(observed.astype(np.uint8), kernel) > 0
    else:
        dilated_predicted = predicted
        dilated_observed = observed

    predicted_near_observed = int((predicted & dilated_observed).sum())
    observed_near_predicted = int((observed & dilated_predicted).sum())
    tolerant_precision = _ratio(predicted_near_observed, predicted_count)
    tolerant_recall = _ratio(observed_near_predicted, observed_count)
    dilated_intersection = int((dilated_predicted & dilated_observed).sum())
    dilated_union = int((dilated_predicted | dilated_observed).sum())

    observed_depth_valid = observed & np.isfinite(observed_depth_m) & (
        observed_depth_m > 0
    )
    depth_overlap = predicted & observed_depth_valid
    absolute_error_m = np.full(observed_depth_m.shape, np.nan, dtype=np.float32)
    absolute_error_m[depth_overlap] = np.abs(
        predicted_depth_m[depth_overlap] - observed_depth_m[depth_overlap]
    )
    finite_errors = absolute_error_m[depth_overlap]
    depth_union = predicted | observed_depth_valid

    metrics: dict[str, float | int] = {
        "predicted_pixels": predicted_count,
        "observed_pixels": observed_count,
        "intersection_pixels": intersection_count,
        "union_pixels": union_count,
        "silhouette_iou": _ratio(intersection_count, union_count),
        "silhouette_precision": precision,
        "silhouette_recall": recall,
        "silhouette_f1": _f1(precision, recall),
        "tolerant_predicted_match_pixels": predicted_near_observed,
        "tolerant_observed_match_pixels": observed_near_predicted,
        "tolerant_precision": tolerant_precision,
        "tolerant_recall": tolerant_recall,
        "tolerant_f1": _f1(tolerant_precision, tolerant_recall),
        "dilated_intersection_pixels": dilated_intersection,
        "dilated_union_pixels": dilated_union,
        "dilated_iou": _ratio(dilated_intersection, dilated_union),
        "observed_depth_pixels": int(observed_depth_valid.sum()),
        "depth_overlap_pixels": int(depth_overlap.sum()),
        "depth_union_pixels": int(depth_union.sum()),
        "depth_error_median_m": (
            float(np.median(finite_errors)) if len(finite_errors) else float("nan")
        ),
        "depth_error_p90_m": (
            float(np.percentile(finite_errors, 90))
            if len(finite_errors)
            else float("nan")
        ),
    }
    for tolerance_m in DEPTH_TOLERANCES_M:
        millimetres = int(round(tolerance_m * 1000.0))
        matched = int((depth_overlap & (absolute_error_m <= tolerance_m)).sum())
        metrics[f"depth_match_{millimetres}mm_pixels"] = matched
        metrics[f"depth_iou_{millimetres}mm"] = _ratio(
            matched, metrics["depth_union_pixels"]
        )
    return metrics, absolute_error_m


def _mask_overlay(
    intensity8: np.ndarray, observed: np.ndarray, predicted: np.ndarray
) -> np.ndarray:
    output = cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)
    colour = output.copy()
    colour[observed] = (190, 70, 190)
    colour[predicted] = (220, 190, 20)
    output = cv2.addWeighted(output, 0.45, colour, 0.55, 0)
    contours, _ = cv2.findContours(
        observed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, (255, 80, 255), 1)
    return output


def _classification_image(
    intensity8: np.ndarray, observed: np.ndarray, predicted: np.ndarray
) -> np.ndarray:
    base = cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)
    output = (base.astype(np.float32) * 0.28).astype(np.uint8)
    true_positive = observed & predicted
    false_positive = predicted & ~observed
    false_negative = observed & ~predicted
    output[true_positive] = (40, 210, 40)
    output[false_positive] = (40, 40, 230)
    output[false_negative] = (230, 100, 30)
    return output


def _depth_colour(
    depth_m: np.ndarray, minimum_m: float, maximum_m: float
) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    scaled = np.zeros(depth_m.shape, dtype=np.uint8)
    if maximum_m > minimum_m:
        clipped = np.clip(depth_m, minimum_m, maximum_m)
        scaled[valid] = np.rint(
            (clipped[valid] - minimum_m) / (maximum_m - minimum_m) * 255.0
        ).astype(np.uint8)
    colour = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    return colour


def _error_colour(error_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(error_m)
    scaled = np.zeros(error_m.shape, dtype=np.uint8)
    scaled[valid] = np.rint(
        np.clip(error_m[valid] / DEPTH_ERROR_DISPLAY_MAX_M, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    colour = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    return colour


def _observed_image(intensity8: np.ndarray, observed: np.ndarray) -> np.ndarray:
    output = cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(
        observed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, (0, 255, 255), 1)
    return output


def _text_panel(
    shape: tuple[int, int], rigid: dict[str, float | int],
    nonrigid: dict[str, float | int]
) -> np.ndarray:
    height, width = shape
    output = np.full((height, width, 3), 24, dtype=np.uint8)
    depth_key = "depth_iou_20mm"
    lines = [
        ("metric", "rigid", "non-rigid", "delta"),
        (
            "silhouette IoU",
            rigid["silhouette_iou"],
            nonrigid["silhouette_iou"],
            nonrigid["silhouette_iou"] - rigid["silhouette_iou"],
        ),
        (
            f"tolerant F1 ({SILHOUETTE_TOLERANCE_PX}px)",
            rigid["tolerant_f1"],
            nonrigid["tolerant_f1"],
            nonrigid["tolerant_f1"] - rigid["tolerant_f1"],
        ),
        (
            "depth IoU (20mm)",
            rigid[depth_key],
            nonrigid[depth_key],
            nonrigid[depth_key] - rigid[depth_key],
        ),
        (
            "depth median (mm)",
            rigid["depth_error_median_m"] * 1000.0,
            nonrigid["depth_error_median_m"] * 1000.0,
            (nonrigid["depth_error_median_m"] - rigid["depth_error_median_m"])
            * 1000.0,
        ),
        (
            "depth p90 (mm)",
            rigid["depth_error_p90_m"] * 1000.0,
            nonrigid["depth_error_p90_m"] * 1000.0,
            (nonrigid["depth_error_p90_m"] - rigid["depth_error_p90_m"])
            * 1000.0,
        ),
    ]
    y = 55
    for index, line in enumerate(lines):
        if index == 0:
            text = f"{line[0]:<23} {line[1]:>9} {line[2]:>11} {line[3]:>9}"
        else:
            text = f"{line[0]:<23} {line[1]:9.4f} {line[2]:11.4f} {line[3]:+9.4f}"
        cv2.putText(
            output, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (225, 225, 225), 1, cv2.LINE_AA
        )
        y += 42
    cv2.putText(
        output, "positive delta is better except depth errors",
        (20, min(y + 20, height - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
        (160, 180, 210), 1, cv2.LINE_AA
    )
    return output


def _individual_visual(
    label: str,
    intensity8: np.ndarray,
    observed: np.ndarray,
    predicted_depth_m: np.ndarray,
    error_m: np.ndarray,
    metrics: dict[str, float | int],
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.isfinite(predicted_depth_m)
    classification = _classification_image(intensity8, observed, predicted)
    title = (
        f"IoU={metrics['silhouette_iou']:.3f}  "
        f"tol-F1={metrics['tolerant_f1']:.3f}  "
        f"depth-IoU20={metrics['depth_iou_20mm']:.3f}"
    )
    panels = [
        ("observed surface outline", _observed_image(intensity8, observed)),
        (f"{label}: projected final cloud", _mask_overlay(
            intensity8, observed, predicted
        )),
        (f"{label}: {title}", classification),
        (
            f"{label}: depth error, median "
            f"{metrics['depth_error_median_m'] * 1000.0:.1f} mm",
            _error_colour(error_m),
        ),
    ]
    return vz.montage(panels, cols=2, tile_w=480), classification


def _comparison_visual(
    frame_id: str,
    intensity8: np.ndarray,
    observed_depth_m: np.ndarray,
    observed: np.ndarray,
    rigid_depth_m: np.ndarray,
    nonrigid_depth_m: np.ndarray,
    rigid_error_m: np.ndarray,
    nonrigid_error_m: np.ndarray,
    rigid_metrics: dict[str, float | int],
    nonrigid_metrics: dict[str, float | int],
) -> np.ndarray:
    observed_values = observed_depth_m[
        observed & np.isfinite(observed_depth_m) & (observed_depth_m > 0)
    ]
    if len(observed_values):
        minimum_m, maximum_m = np.percentile(observed_values, (2, 98))
    else:
        minimum_m, maximum_m = 0.0, 1.0

    rigid_predicted = np.isfinite(rigid_depth_m)
    nonrigid_predicted = np.isfinite(nonrigid_depth_m)
    panels = [
        (f"frame {frame_id}: observed surface", _observed_image(
            intensity8, observed
        )),
        ("rigid projected silhouette", _mask_overlay(
            intensity8, observed, rigid_predicted
        )),
        ("non-rigid projected silhouette", _mask_overlay(
            intensity8, observed, nonrigid_predicted
        )),
        ("observed depth", _depth_colour(
            observed_depth_m, minimum_m, maximum_m
        )),
        ("rigid: green=match red=extra blue=missing",
         _classification_image(intensity8, observed, rigid_predicted)),
        ("non-rigid: green=match red=extra blue=missing",
         _classification_image(intensity8, observed, nonrigid_predicted)),
        ("per-frame numbers", _text_panel(
            intensity8.shape, rigid_metrics, nonrigid_metrics
        )),
        ("rigid absolute depth error", _error_colour(rigid_error_m)),
        ("non-rigid absolute depth error", _error_colour(nonrigid_error_m)),
    ]
    return vz.montage(panels, cols=3, tile_w=400)


def _overall(
    rows: list[dict[str, float | int]], error_arrays: list[np.ndarray]
) -> dict[str, object]:
    def total(key: str) -> int:
        return int(sum(int(row[key]) for row in rows))

    intersection = total("intersection_pixels")
    union = total("union_pixels")
    predicted = total("predicted_pixels")
    observed = total("observed_pixels")
    tolerant_predicted = total("tolerant_predicted_match_pixels")
    tolerant_observed = total("tolerant_observed_match_pixels")
    tolerant_precision = _ratio(tolerant_predicted, predicted)
    tolerant_recall = _ratio(tolerant_observed, observed)

    finite_errors = [
        values[np.isfinite(values)] for values in error_arrays
        if np.isfinite(values).any()
    ]
    all_errors = np.concatenate(finite_errors) if finite_errors else np.array([])

    global_metrics: dict[str, float] = {
        "silhouette_iou": _ratio(intersection, union),
        "silhouette_precision": _ratio(intersection, predicted),
        "silhouette_recall": _ratio(intersection, observed),
        "silhouette_f1": _f1(
            _ratio(intersection, predicted), _ratio(intersection, observed)
        ),
        "tolerant_precision": tolerant_precision,
        "tolerant_recall": tolerant_recall,
        "tolerant_f1": _f1(tolerant_precision, tolerant_recall),
        "dilated_iou": _ratio(
            total("dilated_intersection_pixels"),
            total("dilated_union_pixels"),
        ),
        "depth_error_median_m": (
            float(np.median(all_errors)) if len(all_errors) else float("nan")
        ),
        "depth_error_p90_m": (
            float(np.percentile(all_errors, 90))
            if len(all_errors)
            else float("nan")
        ),
    }
    for tolerance_m in DEPTH_TOLERANCES_M:
        millimetres = int(round(tolerance_m * 1000.0))
        global_metrics[f"depth_iou_{millimetres}mm"] = _ratio(
            total(f"depth_match_{millimetres}mm_pixels"),
            total("depth_union_pixels"),
        )

    comparison_keys = [
        "silhouette_iou", "silhouette_precision", "silhouette_recall",
        "silhouette_f1", "tolerant_precision", "tolerant_recall",
        "tolerant_f1", "dilated_iou", "depth_error_median_m",
        "depth_error_p90_m",
    ] + [
        f"depth_iou_{int(round(value * 1000.0))}mm"
        for value in DEPTH_TOLERANCES_M
    ]
    per_frame = {}
    for key in comparison_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        per_frame[key] = {
            "mean": float(np.mean(finite)) if len(finite) else float("nan"),
            "median": float(np.median(finite)) if len(finite) else float("nan"),
            "minimum": float(np.min(finite)) if len(finite) else float("nan"),
            "maximum": float(np.max(finite)) if len(finite) else float("nan"),
        }
    return {"global": global_metrics, "per_frame": per_frame}


def _save_metrics_plot(
    frame_ids: list[str],
    rigid_rows: list[dict[str, float | int]],
    nonrigid_rows: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    positions = np.arange(len(frame_ids))
    width = 0.38
    plots = [
        ("silhouette_iou", "Silhouette IoU", 1.0),
        ("tolerant_f1", f"Tolerant F1 ({SILHOUETTE_TOLERANCE_PX}px)", 1.0),
        ("depth_iou_20mm", "Depth-aware IoU (20 mm)", 1.0),
        ("depth_error_p90_m", "Depth error p90 (mm)", None),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(14, 8))
    for axis, (key, title, maximum) in zip(axes.ravel(), plots):
        rigid_values = np.array([float(row[key]) for row in rigid_rows])
        nonrigid_values = np.array([float(row[key]) for row in nonrigid_rows])
        if key.endswith("_m"):
            rigid_values *= 1000.0
            nonrigid_values *= 1000.0
        axis.bar(positions - width / 2, rigid_values, width, label="rigid")
        axis.bar(positions + width / 2, nonrigid_values, width, label="non-rigid")
        axis.set_title(title)
        axis.set_xticks(positions, frame_ids)
        axis.set_xlabel("frame")
        if maximum is not None:
            axis.set_ylim(0.0, maximum)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _write_csv(
    frame_ids: list[str],
    rigid_rows: list[dict[str, float | int]],
    nonrigid_rows: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    fields = ["frame_id", "method"] + list(rigid_rows[0])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, rows in (("rigid", rigid_rows), ("nonrigid", nonrigid_rows)):
            for frame_id, metrics in zip(frame_ids, rows):
                writer.writerow({"frame_id": frame_id, "method": method, **metrics})


def _metric_delta(
    rigid: dict[str, float], nonrigid: dict[str, float]
) -> dict[str, float]:
    keys = sorted(set(rigid) & set(nonrigid))
    return {key: float(nonrigid[key] - rigid[key]) for key in keys}

def _json_safe(value):
    """Replace non-finite NumPy/Python scalars so reports remain valid JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def run() -> Path:
    rigid_path = RIGID_PLY.resolve()
    nonrigid_path = (
        NONRIGID_PLY.resolve()
        if NONRIGID_PLY is not None
        else _default_nonrigid_path(rigid_path).resolve()
    )
    poses_path = (
        POSES_PATH.resolve()
        if POSES_PATH is not None
        else _default_poses_path(rigid_path).resolve()
    )
    manifest_path = (
        FRAMES_MANIFEST_PATH.resolve()
        if FRAMES_MANIFEST_PATH is not None
        else _default_manifest_path(rigid_path).resolve()
    )
    output_dir = (
        OUTPUT_DIR.resolve()
        if OUTPUT_DIR is not None
        else _default_output_dir(rigid_path).resolve()
    )
    mask_dir = MASK_DIR.resolve() if MASK_DIR is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rigid").mkdir(exist_ok=True)
    (output_dir / "nonrigid").mkdir(exist_ok=True)
    (output_dir / "comparison").mkdir(exist_ok=True)

    frame_ids, final_poses, intrinsics, frame_paths = _read_metadata(
        poses_path, manifest_path
    )
    rigid_points, nonrigid_points, ranges = _load_cloud_pair(
        rigid_path, nonrigid_path, frame_ids
    )
    print(
        f"loaded {len(rigid_points):,} corresponding rigid/non-rigid points "
        f"from {len(frame_ids)} frames"
    )
    print(
        "observed mask source: "
        + (str(mask_dir) if mask_dir is not None else "valid depth pixels")
    )
    print(
        "source-frame points: "
        + ("excluded" if EXCLUDE_SOURCE_FRAME_POINTS else "included")
    )

    rigid_rows: list[dict[str, float | int]] = []
    nonrigid_rows: list[dict[str, float | int]] = []
    rigid_errors: list[np.ndarray] = []
    nonrigid_errors: list[np.ndarray] = []
    mask_sources: dict[str, str] = {}
    rigid_overview = []
    nonrigid_overview = []
    paired_overview = []

    for index, frame_id in enumerate(frame_ids):
        print(f"frame {frame_id}: projecting rigid and non-rigid final clouds")
        observed_depth_m, intensity8 = _read_frame(frame_paths[index])
        observed, mask_source = _read_observed_mask(
            frame_id, observed_depth_m, mask_dir
        )
        mask_sources[frame_id] = mask_source
        excluded_range = ranges[index] if EXCLUDE_SOURCE_FRAME_POINTS else None

        rigid_depth_m = _render_depth(
            rigid_points, final_poses[index], intrinsics[index],
            observed_depth_m.shape, excluded_range
        )
        nonrigid_depth_m = _render_depth(
            nonrigid_points, final_poses[index], intrinsics[index],
            observed_depth_m.shape, excluded_range
        )
        rigid_metrics, rigid_error_m = _evaluate_projection(
            rigid_depth_m, observed_depth_m, observed
        )
        nonrigid_metrics, nonrigid_error_m = _evaluate_projection(
            nonrigid_depth_m, observed_depth_m, observed
        )
        rigid_rows.append(rigid_metrics)
        nonrigid_rows.append(nonrigid_metrics)
        rigid_errors.append(rigid_error_m)
        nonrigid_errors.append(nonrigid_error_m)

        rigid_image, rigid_classification = _individual_visual(
            "rigid", intensity8, observed, rigid_depth_m,
            rigid_error_m, rigid_metrics
        )
        nonrigid_image, nonrigid_classification = _individual_visual(
            "non-rigid", intensity8, observed, nonrigid_depth_m,
            nonrigid_error_m, nonrigid_metrics
        )
        comparison_image = _comparison_visual(
            frame_id, intensity8, observed_depth_m, observed,
            rigid_depth_m, nonrigid_depth_m, rigid_error_m,
            nonrigid_error_m, rigid_metrics, nonrigid_metrics
        )

        filenames = {
            output_dir / "rigid" / f"frame_{frame_id}.png": rigid_image,
            output_dir / "nonrigid" / f"frame_{frame_id}.png": nonrigid_image,
            output_dir / "comparison" / f"frame_{frame_id}.png": comparison_image,
        }
        for path, image in filenames.items():
            if not cv2.imwrite(str(path), image):
                raise OSError(f"failed to save visualization: {path}")

        if index < MAX_MONTAGE_FRAMES:
            rigid_overview.append((f"frame {frame_id}: rigid", rigid_classification))
            nonrigid_overview.append(
                (f"frame {frame_id}: non-rigid", nonrigid_classification)
            )
            paired_overview.extend(
                [
                    (f"frame {frame_id}: rigid", rigid_classification),
                    (f"frame {frame_id}: non-rigid", nonrigid_classification),
                ]
            )

    rigid_summary = _overall(rigid_rows, rigid_errors)
    nonrigid_summary = _overall(nonrigid_rows, nonrigid_errors)
    comparison = _metric_delta(
        rigid_summary["global"], nonrigid_summary["global"]
    )

    report = {
        "rigid_ply": str(rigid_path),
        "nonrigid_ply": str(nonrigid_path),
        "poses_path": str(poses_path),
        "frames_manifest": str(manifest_path),
        "frame_ids": frame_ids,
        "evaluation": {
            "scores_only_final_point_clouds": True,
            "source_frame_points_excluded": EXCLUDE_SOURCE_FRAME_POINTS,
            "mask_mode": "binary_masks" if mask_dir is not None else "valid_depth",
            "mask_sources": mask_sources,
            "render_point_radius_px": RENDER_POINT_RADIUS_PX,
            "silhouette_tolerance_px": SILHOUETTE_TOLERANCE_PX,
            "depth_tolerances_m": list(DEPTH_TOLERANCES_M),
        },
        "rigid": {"frames": rigid_rows, "summary": rigid_summary},
        "nonrigid": {"frames": nonrigid_rows, "summary": nonrigid_summary},
        "nonrigid_minus_rigid_global": comparison,
    }
    report_path = output_dir / "projection_fit_report.json"
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        frame_ids, rigid_rows, nonrigid_rows,
        output_dir / "projection_fit_metrics.csv"
    )
    _save_metrics_plot(
        frame_ids, rigid_rows, nonrigid_rows,
        output_dir / "metrics_comparison.png"
    )
    cv2.imwrite(
        str(output_dir / "rigid_all_frames.png"),
        vz.montage(rigid_overview, cols=2, tile_w=420),
    )
    cv2.imwrite(
        str(output_dir / "nonrigid_all_frames.png"),
        vz.montage(nonrigid_overview, cols=2, tile_w=420),
    )
    cv2.imwrite(
        str(output_dir / "rigid_vs_nonrigid_all_frames.png"),
        vz.montage(paired_overview, cols=2, tile_w=420),
    )

    rigid_global = rigid_summary["global"]
    nonrigid_global = nonrigid_summary["global"]
    print("")
    print("final-cloud projection fit")
    print("metric                       rigid   non-rigid       delta")
    for key, scale in (
        ("silhouette_iou", 1.0),
        ("tolerant_f1", 1.0),
        ("depth_iou_20mm", 1.0),
        ("depth_error_median_m", 1000.0),
        ("depth_error_p90_m", 1000.0),
    ):
        label = key + (" (mm)" if scale == 1000.0 else "")
        rigid_value = rigid_global[key] * scale
        nonrigid_value = nonrigid_global[key] * scale
        print(
            f"{label:<28} {rigid_value:8.4f} "
            f"{nonrigid_value:11.4f} {nonrigid_value - rigid_value:+11.4f}"
        )
    print(f"outputs -> {output_dir}")
    return report_path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
