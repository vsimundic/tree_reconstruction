#!/usr/bin/env python3
"""Compare connected-region SAM prompts with whole-image SAM segmentation.

The production detector is not modified.  This experiment runs it as the
baseline, then emulates SAM automatic-mask generation with a uniform point grid
covering the complete image.  Both methods use the same rectangle scoring, NMS,
and 3-D marker-model registration.  Manual marker centers are evaluation only.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

TREE_RECONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
if str(TREE_RECONSTRUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(TREE_RECONSTRUCTION_DIR))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import depth_regions as dr
import marker_detection_sam3 as baseline
import marker_registration as reg
import sam3_client
import viz_utils as vz


# ---------------------------------------------------------------------------
# Configuration -- edit these variables, then run this file without arguments.
# ---------------------------------------------------------------------------
FRAME_DIR = Path(
    "data/pears/pears_filtered_clean/A/tree_0_V_0000/marker_captures/depth")
FRAME_IDS = tuple(range(8))
OUTPUT_DIR = Path("data/marker_detection_comparison/A/tree_0_V_0000")

# One positive point per grid cell.  The grid is independent of image content;
# unlike the baseline, it does not use connected regions to select prompts.
GRID_SPACING_PX = 48
SAM_MULTIMASK = True
MIN_SAM_SCORE = 0.0
MAX_MASK_AREA_FRAC = dr.MAX_AREA_FRAC
MAX_3D_CANDIDATES = dr.TOP_K
NMS_CENTER_DISTANCE_PX = dr.NMS_DIST_PX
MAX_MASKS_IN_OVERLAY = 40
OVERWRITE = True


def _grid_points(shape: tuple[int, int], spacing_px: int) -> list[tuple[int, int]]:
    height, width = shape
    offset = spacing_px // 2
    return [
        (x, y)
        for y in range(offset, height, spacing_px)
        for x in range(offset, width, spacing_px)
    ]


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def _whole_image_candidates(
        image: np.ndarray, shape: tuple[int, int]
) -> tuple[list[dict], list[tuple[int, int]], dict[str, int]]:
    """Prompt SAM on a content-independent grid and score every returned mask."""
    height, width = shape
    prompts = _grid_points(shape, GRID_SPACING_PX)
    hypotheses = []
    mask_count = 0
    area_pass_count = 0

    for prompt_index, prompt in enumerate(prompts, start=1):
        output = sam3_client.segment(
            image, points=[prompt], point_labels=[1], multimask=SAM_MULTIMASK)
        masks = output["masks"]
        scores = np.asarray(output["scores"]).ravel()
        mask_count += len(masks)
        for mask, sam_score in zip(masks, scores):
            if float(sam_score) < MIN_SAM_SCORE:
                continue
            mask_bool = mask > 0
            area = int(mask_bool.sum())
            if not (dr.MIN_AREA_PX <= area <= MAX_MASK_AREA_FRAC * height * width):
                continue
            contour = _largest_contour(mask_bool)
            if contour is None:
                continue
            contour_area = cv2.contourArea(contour)
            if not (dr.MIN_AREA_PX <= contour_area
                    <= MAX_MASK_AREA_FRAC * height * width):
                continue
            area_pass_count += 1
            metrics = dr.score_rectangle_shape(contour)
            if metrics is None:
                continue

            shape_score = float(metrics["score"])
            metrics.update(
                source="sam3_whole_image",
                refined_mask=mask_bool,
                prompt=prompt,
                prompt_index=prompt_index,
                sam_score=float(sam_score),
                shape_score=shape_score,
                # Complete, confident masks should beat nearby partial masks.
                score=shape_score * float(sam_score),
            )
            hypotheses.append(metrics)

    hypotheses.sort(key=lambda item: -item["score"])
    counts = {
        "sam_calls": len(prompts),
        "returned_masks": mask_count,
        "area_pass_masks": area_pass_count,
        "rectangle_masks": len(hypotheses),
    }
    return hypotheses, prompts, counts


def _draw_grid(intensity8: np.ndarray, prompts: list[tuple[int, int]]) -> np.ndarray:
    output = cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)
    for point in prompts:
        cv2.circle(output, point, 2, (0, 220, 255), -1)
    return output


def _overlay_masks(intensity8: np.ndarray, hypotheses: list[dict]) -> np.ndarray:
    output = cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)
    overlay = output.copy()
    colours = baseline._region_colors(min(len(hypotheses), MAX_MASKS_IN_OVERLAY), seed=7)
    for hypothesis, colour in zip(hypotheses[:MAX_MASKS_IN_OVERLAY], colours):
        overlay[hypothesis["refined_mask"]] = colour
    return cv2.addWeighted(overlay, 0.45, output, 0.55, 0)


def _draw_scored_masks(intensity8: np.ndarray, hypotheses: list[dict]) -> np.ndarray:
    output = cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)
    for hypothesis in hypotheses[:MAX_MASKS_IN_OVERLAY]:
        cv2.polylines(output, [hypothesis["approx"]], True, (40, 220, 40), 1)
        x, y = hypothesis["bbox"][:2]
        cv2.putText(
            output, f"{hypothesis['score']:.2f}", (int(x), max(int(y) - 3, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 220, 40), 1, cv2.LINE_AA)
    return output


def run_whole_image_pipeline(
        depth_m: np.ndarray, intensity8: np.ndarray, K: np.ndarray
) -> tuple[list[tuple[str, np.ndarray]], list[dict], list[dict], dict[str, int]]:
    image = sam3_client.to_uint8_rgb(intensity8)
    hypotheses, prompts, counts = _whole_image_candidates(image, depth_m.shape)
    kept = dr.nms_by_center(
        hypotheses, NMS_CENTER_DISTANCE_PX, MAX_3D_CANDIDATES)
    results = reg.verify_candidates(kept, depth_m, K)

    panels = [
        ("1. intensity (brightened)", intensity8),
        ("2. depth", vz.colorize(depth_m, valid=depth_m > 0)),
        (f"3. whole-image SAM grid ({len(prompts)} prompts)",
         _draw_grid(intensity8, prompts)),
        (f"4. rectangle-like SAM masks ({len(hypotheses)})",
         _overlay_masks(intensity8, hypotheses)),
        ("5. whole-image masks scored by SAM x rectangle quality",
         _draw_scored_masks(intensity8, hypotheses)),
        (f"6. final hypotheses after NMS ({len(kept)})",
         dr.draw_candidates(intensity8, kept) if kept else
         cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)),
        (f"7. 3-D registration verdicts ({len(results)})",
         reg.draw_verified(intensity8, results, K)),
        ("8. best-fit model reprojection",
         reg.project_model_overlay(intensity8, results[0], K) if results else
         cv2.cvtColor(intensity8, cv2.COLOR_GRAY2BGR)),
    ]
    counts["kept_2d_candidates"] = len(kept)
    counts["registered_candidates"] = len(results)
    return panels, kept, results, counts


def _accepted(results: list[dict]) -> bool:
    return bool(results and results[0]["accepted"])


def _registered_center_px(result: dict, K: np.ndarray) -> tuple[float, float] | None:
    center = np.asarray(result["center"])
    if not np.isfinite(center).all() or center[2] <= 0:
        return None
    return (
        float(center[0] / center[2] * K[0, 0] + K[0, 2]),
        float(center[1] / center[2] * K[1, 1] + K[1, 2]),
    )


def _evaluation(results: list[dict], K: np.ndarray,
                ground_truth: tuple[int, int] | None) -> dict:
    accepted = _accepted(results)
    if not results:
        return {
            "accepted": False,
            "registered_center_px": None,
            "registered_center_error_px": None,
            "candidate_center_error_px": None,
            "fitness": None,
            "relief_fitness": None,
            "rmse_m": None,
        }
    top = results[0]
    registered_center = _registered_center_px(top, K)
    candidate = top["cand"]
    x, y, width, height = candidate["bbox"]
    candidate_center = (x + width / 2.0, y + height / 2.0)

    def error(point):
        if point is None or ground_truth is None:
            return None
        return float(np.hypot(point[0] - ground_truth[0], point[1] - ground_truth[1]))

    return {
        "accepted": accepted,
        "registered_center_px": registered_center,
        "registered_center_error_px": error(registered_center),
        "candidate_center_error_px": error(candidate_center),
        "fitness": float(top["fitness"]),
        "relief_fitness": float(top["relief_fitness"]),
        "rmse_m": float(top["rmse"]),
    }


def _pose_difference(baseline_results: list[dict], whole_results: list[dict]) -> dict:
    if not (_accepted(baseline_results) and _accepted(whole_results)):
        return {"center_difference_m": None, "rotation_difference_deg": None}
    first = baseline_results[0]
    second = whole_results[0]
    center_difference = float(np.linalg.norm(first["center"] - second["center"]))
    relative_rotation = first["transform"][:3, :3].T @ second["transform"][:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "center_difference_m": center_difference,
        "rotation_difference_deg": float(np.degrees(np.arccos(cosine))),
    }


def _with_ground_truth(
        image: np.ndarray, ground_truth: tuple[int, int] | None) -> np.ndarray:
    output = image.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    if ground_truth is not None:
        cv2.drawMarker(output, ground_truth, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(
            output, "manual eval", (ground_truth[0] + 7, ground_truth[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return output


def _save_panels(panels: list[tuple[str, np.ndarray]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (title, image) in enumerate(panels, start=1):
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        path = output_dir / f"{index:02d}_{slug}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"failed to save {path}")


def _frame_comparison(
        frame_id: int, intensity8: np.ndarray,
        baseline_panels: list[tuple[str, np.ndarray]],
        whole_panels: list[tuple[str, np.ndarray]],
        ground_truth: tuple[int, int] | None) -> np.ndarray:
    panels = [
        (f"frame {frame_id}: input + evaluation point",
         _with_ground_truth(intensity8, ground_truth)),
        ("connected regions: SAM prompts", baseline_panels[7][1]),
        ("whole image: SAM grid prompts", whole_panels[2][1]),
        ("connected regions: final 2-D candidates",
         _with_ground_truth(baseline_panels[10][1], ground_truth)),
        ("whole image: final 2-D candidates",
         _with_ground_truth(whole_panels[5][1], ground_truth)),
        ("connected regions: 3-D verdicts",
         _with_ground_truth(baseline_panels[11][1], ground_truth)),
        ("whole image: 3-D verdicts",
         _with_ground_truth(whole_panels[6][1], ground_truth)),
        ("connected regions: best model fit",
         _with_ground_truth(baseline_panels[12][1], ground_truth)),
        ("whole image: best model fit",
         _with_ground_truth(whole_panels[7][1], ground_truth)),
    ]
    return vz.montage(panels, cols=3, tile_w=420)


def _save_summary_plot(rows: list[dict], output_path: Path) -> None:
    frame_labels = [str(row["frame_id"]) for row in rows]
    positions = np.arange(len(rows))
    width = 0.36
    baseline_success = [int(row["connected_regions"]["accepted"]) for row in rows]
    whole_success = [int(row["whole_image_sam"]["accepted"]) for row in rows]
    baseline_error = [
        row["connected_regions"]["registered_center_error_px"] or np.nan for row in rows]
    whole_error = [
        row["whole_image_sam"]["registered_center_error_px"] or np.nan for row in rows]
    baseline_time = [row["connected_regions"]["runtime_s"] for row in rows]
    whole_time = [row["whole_image_sam"]["runtime_s"] for row in rows]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(positions - width / 2, baseline_success, width, label="connected regions")
    axes[0].bar(positions + width / 2, whole_success, width, label="whole-image SAM")
    axes[0].set_title("Accepted 3-D marker")
    axes[0].set_ylim(0, 1.15)
    axes[0].set_yticks((0, 1))
    axes[0].set_xticks(positions, frame_labels)
    axes[0].set_xlabel("frame")
    axes[0].legend(fontsize=8)

    axes[1].bar(positions - width / 2, baseline_error, width)
    axes[1].bar(positions + width / 2, whole_error, width)
    axes[1].set_title("Registered center error")
    axes[1].set_ylabel("pixels (lower is better)")
    axes[1].set_xticks(positions, frame_labels)
    axes[1].set_xlabel("frame")

    axes[2].bar(positions - width / 2, baseline_time, width)
    axes[2].bar(positions + width / 2, whole_time, width)
    axes[2].set_title("Runtime")
    axes[2].set_ylabel("seconds")
    axes[2].set_xticks(positions, frame_labels)
    axes[2].set_xlabel("frame")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _serializable_row(row: dict) -> dict:
    """The result rows already contain only JSON-safe scalars/lists."""
    return row


def run() -> Path:
    frame_dir = FRAME_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    health = sam3_client.health()
    print(f"SAM service: device={health.get('device')} "
          f"mask_input_hw={health.get('mask_input_hw')}")

    rows = []
    final_candidate_panels = []
    verdict_panels = []
    for frame_id in FRAME_IDS:
        frame_output = output_dir / f"frame_{frame_id:04d}"
        comparison_path = frame_output / "comparison.png"
        if comparison_path.exists() and not OVERWRITE:
            print(f"frame {frame_id}: comparison already exists -- skipped")
            continue

        print(f"frame {frame_id}: connected-region baseline", flush=True)
        intensity, intensity8, depth_m, K = dr.load_frame(str(frame_dir), frame_id)
        _, baseline_proposals = dr.sweep_region_proposals(
            depth_m, intensity, max_regions=baseline.SAM3_MAX_REGIONS_PER_FRAME)
        started = time.perf_counter()
        baseline_panels, baseline_hypotheses, baseline_results = baseline.run_pipeline(
            depth_m, intensity, intensity8, K)
        baseline_runtime = time.perf_counter() - started

        print(f"frame {frame_id}: whole-image SAM ({len(_grid_points(depth_m.shape, GRID_SPACING_PX))} calls)",
              flush=True)
        started = time.perf_counter()
        whole_panels, whole_hypotheses, whole_results, whole_counts = (
            run_whole_image_pipeline(depth_m, intensity8, K))
        whole_runtime = time.perf_counter() - started

        ground_truth = baseline.get_ground_truth_point(str(frame_dir), frame_id)
        baseline_eval = _evaluation(baseline_results, K, ground_truth)
        whole_eval = _evaluation(whole_results, K, ground_truth)
        baseline_eval.update(
            runtime_s=float(baseline_runtime),
            sam_calls=len(baseline_proposals),
            kept_2d_candidates=len(baseline_hypotheses),
            registered_candidates=len(baseline_results),
        )
        whole_eval.update(runtime_s=float(whole_runtime), **whole_counts)
        row = {
            "frame_id": frame_id,
            "manual_evaluation_center_px": ground_truth,
            "connected_regions": baseline_eval,
            "whole_image_sam": whole_eval,
            "accepted_pose_difference": _pose_difference(
                baseline_results, whole_results),
        }
        rows.append(row)

        _save_panels(baseline_panels, frame_output / "connected_regions")
        _save_panels(whole_panels, frame_output / "whole_image_sam")
        comparison = _frame_comparison(
            frame_id, intensity8, baseline_panels, whole_panels, ground_truth)
        frame_output.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(comparison_path), comparison):
            raise OSError(f"failed to save {comparison_path}")

        final_candidate_panels.extend([
            (f"frame {frame_id}: connected regions",
             _with_ground_truth(baseline_panels[10][1], ground_truth)),
            (f"frame {frame_id}: whole-image SAM",
             _with_ground_truth(whole_panels[5][1], ground_truth)),
        ])
        verdict_panels.extend([
            (f"frame {frame_id}: connected regions",
             _with_ground_truth(baseline_panels[11][1], ground_truth)),
            (f"frame {frame_id}: whole-image SAM",
             _with_ground_truth(whole_panels[6][1], ground_truth)),
        ])
        print(
            f"frame {frame_id}: accepted connected={baseline_eval['accepted']} "
            f"whole={whole_eval['accepted']}; runtime "
            f"{baseline_runtime:.1f}s vs {whole_runtime:.1f}s",
            flush=True)

    if not rows:
        raise RuntimeError("no frames were processed")

    candidates_montage = vz.montage(final_candidate_panels, cols=2, tile_w=520)
    verdicts_montage = vz.montage(verdict_panels, cols=2, tile_w=520)
    cv2.imwrite(str(output_dir / "all_frames_2d_candidates.png"), candidates_montage)
    cv2.imwrite(str(output_dir / "all_frames_3d_verdicts.png"), verdicts_montage)
    _save_summary_plot(rows, output_dir / "summary_metrics.png")

    summary = {
        "frame_dir": str(frame_dir),
        "frame_ids": list(FRAME_IDS),
        "whole_image_grid_spacing_px": GRID_SPACING_PX,
        "manual_annotations_used_as_input": False,
        "rows": [_serializable_row(row) for row in rows],
        "totals": {
            "connected_regions_accepted": sum(
                row["connected_regions"]["accepted"] for row in rows),
            "whole_image_sam_accepted": sum(
                row["whole_image_sam"]["accepted"] for row in rows),
            "connected_regions_runtime_s": sum(
                row["connected_regions"]["runtime_s"] for row in rows),
            "whole_image_sam_runtime_s": sum(
                row["whole_image_sam"]["runtime_s"] for row in rows),
        },
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"comparison outputs -> {output_dir}")
    return report_path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
