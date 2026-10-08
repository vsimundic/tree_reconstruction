#!/usr/bin/env python3
"""Build presentation figures from saved reconstruction outputs.

This script does not run or modify the reconstruction. It composes existing
diagnostic images and draws camera/pose-graph diagrams from the saved pose
sidecar. Edit the variables below, then run the file without arguments.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Configuration -- edit these variables, then run this file without arguments.
# ---------------------------------------------------------------------------
RECONSTRUCTION_DIR = Path(
    "data/pears/pears_reconstructed/B/tree_0_V_0000")
CAPTURE_NAME = "tree_0_V_0000"
MARKER_EXAMPLE_DIR = Path(
    "data/marker_detection_comparison/A/tree_0_V_0000/"
    "frame_0000/connected_regions")
OUTPUT_DIR = RECONSTRUCTION_DIR / "presentation_figures"

WIDTH = 1920
HEIGHT = 1080
BACKGROUND = (13, 20, 32)
PANEL = (24, 35, 52)
WHITE = (240, 245, 250)
MUTED = (166, 181, 199)
CYAN = (61, 201, 236)
GREEN = (84, 211, 140)
ORANGE = (255, 166, 76)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def _canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), title, font=_font(48, True), fill=WHITE)
    if subtitle:
        draw.text((72, 108), subtitle, font=_font(25), fill=MUTED)
    return image, draw


def _fit(source: Path | Image.Image, size: tuple[int, int]) -> Image.Image:
    image = Image.open(source).convert("RGB") if isinstance(source, Path) else source.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    result = Image.new("RGB", size, PANEL)
    result.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return result


def _panel(canvas: Image.Image, source: Path | Image.Image, box: tuple[int, int, int, int],
           label: str, accent: tuple[int, int, int] = CYAN) -> None:
    x, y, w, h = box
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=PANEL,
                           outline=accent, width=3)
    label_h = 52
    fitted = _fit(source, (w - 16, h - label_h - 16))
    canvas.paste(fitted, (x + 8, y + label_h))
    draw.text((x + 18, y + 12), label, font=_font(24, True), fill=accent)


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int],
           colour: tuple[int, int, int] = MUTED) -> None:
    draw.line((start, end), fill=colour, width=8)
    x, y = end
    draw.polygon(((x, y), (x - 22, y - 13), (x - 22, y + 13)), fill=colour)


def _save(image: Image.Image, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    image.save(path, quality=95)
    print(path)
    return path


def _marker_paths() -> dict[str, Path]:
    patterns = {
        "Depth boundaries": "03_3_boundaries_*.png",
        "Connected regions": "05_5_connected_components_*.png",
        "Merged regions": "06_6_merged_components_*.png",
        "SAM refinement": "10_10_sam3_refined_*.png",
        "3D fit verdicts": "12_12_registration_verdicts_*.png",
        "Accepted marker": "13_13_best_fit_model_reprojection.png",
    }
    paths = {}
    for label, pattern in patterns.items():
        matches = sorted(MARKER_EXAMPLE_DIR.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"missing marker image: {MARKER_EXAMPLE_DIR / pattern}")
        paths[label] = matches[0]
    return paths


def _comparison_paths() -> dict[str, Path]:
    nonrigid = RECONSTRUCTION_DIR / "non_rigid_comparison"
    projection = (RECONSTRUCTION_DIR / "projection_fit_comparison" /
                  f"{CAPTURE_NAME}_merged_scene")
    return {
        "front": nonrigid / f"{CAPTURE_NAME}_merged_scene_comparison_front.png",
        "side": nonrigid / f"{CAPTURE_NAME}_merged_scene_comparison_side.png",
        "mobility": nonrigid / f"{CAPTURE_NAME}_merged_scene_mobility_front.png",
        "movement": nonrigid / f"{CAPTURE_NAME}_merged_scene_movement_front.png",
        "metrics": projection / "metrics_comparison.png",
        "projection": projection / "comparison" / "frame_0.png",
        "all_views": projection / "rigid_vs_nonrigid_all_frames.png",
    }


def _load_poses() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    path = RECONSTRUCTION_DIR / f"{CAPTURE_NAME}_merged.poses.npz"
    with np.load(path, allow_pickle=False) as poses:
        frame_ids = [str(value) for value in poses["frame_ids"]]
        final = np.asarray(poses["final_poses"], dtype=np.float64)
        marker = np.asarray(poses["T_marker_cam"], dtype=np.float64)
    marker_ref = marker[0]
    initial = np.stack([marker_ref @ np.linalg.inv(value) for value in marker])
    marker_center = marker_ref[:3, 3]
    return frame_ids, initial, final, marker_center


def make_slide_1(marker: dict[str, Path], figures: dict[str, Path]) -> None:
    image, draw = _canvas("Reconstruction method — overview",
                          "Every thumbnail below is produced by the project code")
    boxes = [
        (75, 260, 310, 500, marker["Depth boundaries"], "Depth views", CYAN),
        (455, 260, 310, 500, marker["Accepted marker"], "Marker pose", CYAN),
        (835, 260, 310, 500, figures["front"], "Rigid alignment", GREEN),
        (1215, 260, 310, 500, figures["mobility"], "Movable tips", ORANGE),
        (1595, 260, 250, 500, figures["metrics"], "Evaluation", GREEN),
    ]
    for x, y, w, h, source, label, colour in boxes:
        _panel(image, source, (x, y, w, h), label, colour)
    for first, second in zip(boxes, boxes[1:]):
        _arrow(draw, (first[0] + first[2] + 12, 510), (second[0] - 12, 510))
    draw.text((650, 835), "marker → robust ICP → pose graph → optional branch-tip refinement",
              font=_font(31, True), fill=WHITE)
    _save(image, "slide_01_method_overview.png")


def make_slide_2(marker: dict[str, Path]) -> None:
    image, _ = _canvas("Marker candidates and 3D verification",
                       "Depth-continuity regions are refined by SAM, then checked against the known 3D marker")
    positions = [(65, 175), (680, 175), (1295, 175), (65, 610), (680, 610), (1295, 610)]
    for (label, path), (x, y) in zip(marker.items(), positions):
        _panel(image, path, (x, y, 560, 370), label, CYAN if y < 600 else GREEN)
    _save(image, "slide_02_marker_detection_and_fit.png")


def _pose_plot(frame_ids: list[str], initial: np.ndarray, final: np.ndarray,
               marker_center: np.ndarray) -> Image.Image:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 7.2), dpi=120, facecolor=np.array(BACKGROUND) / 255)
    for index, (poses, title, colour) in enumerate((
            (initial, "Marker-based initial poses", "#3dc9ec"),
            (final, "After pose-graph optimization", "#54d38c")), start=1):
        axis = fig.add_subplot(1, 2, index, projection="3d")
        centers = poses[:, :3, 3]
        axis.plot(centers[:, 0], centers[:, 1], centers[:, 2], color=colour,
                  linewidth=2, alpha=0.8)
        axis.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color=colour, s=55)
        for fid, center, pose in zip(frame_ids, centers, poses):
            axis.text(*center, fid, color="white", fontsize=9)
            forward = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
            axis.quiver(*center, *(0.12 * forward), color=colour, linewidth=1.2)
        axis.scatter(*marker_center, color="#ffa64c", marker="s", s=100,
                     label="shared marker")
        axis.set_title(title, fontsize=15, pad=18)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.legend(loc="upper left")
        axis.set_box_aspect((1, 1, 0.7))
        axis.grid(alpha=0.18)
    fig.tight_layout(pad=2)
    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(array)


def make_slide_3(frame_ids: list[str], initial: np.ndarray, final: np.ndarray,
                 marker_center: np.ndarray) -> None:
    plot = _pose_plot(frame_ids, initial, final, marker_center)
    image, draw = _canvas("Initial camera alignment from the shared marker",
                          "Camera axes and positions come from the saved pose sidecar")
    image.paste(_fit(plot, (1780, 790)), (70, 175))
    draw.text((610, 990), "The first accepted camera is the fixed reference frame",
              font=_font(28, True), fill=CYAN)
    _save(image, "slide_03_marker_camera_poses.png")


def make_slide_4(frame_ids: list[str], final: np.ndarray) -> None:
    centers = final[:, :3, 3]
    centered = centers - centers.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:2].T
    plt.style.use("dark_background")
    fig, axis = plt.subplots(figsize=(16, 7.4), dpi=120,
                             facecolor=np.array(BACKGROUND) / 255)
    axis.set_facecolor(np.array(PANEL) / 255)
    for i in range(len(frame_ids)):
        for j in range(i + 1, min(len(frame_ids), i + 4)):
            consecutive = j == i + 1
            axis.plot(projected[[i, j], 0], projected[[i, j], 1],
                      color="#3dc9ec" if consecutive else "#ffa64c",
                      linewidth=3 if consecutive else 1.7,
                      linestyle="-" if consecutive else "--", alpha=0.85)
    axis.scatter(projected[:, 0], projected[:, 1], s=180, color="#54d38c",
                 edgecolor="white", linewidth=1.5, zorder=4)
    for fid, point in zip(frame_ids, projected):
        axis.text(point[0], point[1], fid, ha="center", va="center",
                  color="#0d1420", fontsize=11, fontweight="bold", zorder=5)
    axis.plot([], [], color="#3dc9ec", linewidth=3, label="consecutive-frame ICP")
    axis.plot([], [], color="#ffa64c", linestyle="--", linewidth=2,
              label="additional neighbour-pair ICP")
    axis.legend(loc="upper right", fontsize=12)
    axis.set_title("Neighbour-pair constraints used by the global pose graph", fontsize=17)
    axis.set_xlabel("camera-layout axis 1")
    axis.set_ylabel("camera-layout axis 2")
    axis.grid(alpha=0.15)
    axis.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(pad=2)
    fig.canvas.draw()
    graph = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

    image, draw = _canvas("Pairwise robust ICP builds the global pose graph",
                          "Edges are relative-pose measurements; the graph adjusts all camera poses together")
    image.paste(_fit(graph, (1280, 805)), (55, 175))
    x = 1380
    notes = [
        ("Point-to-plane ICP", WHITE),
        ("Coarse: 50 mm", CYAN),
        ("Fine: 15 mm", GREEN),
        ("Tukey width: 7.5 mm", ORANGE),
        ("Pairs up to 3 frames apart", WHITE),
        ("Reference camera fixed", WHITE),
    ]
    draw.rounded_rectangle((1350, 250, 1870, 795), radius=22, fill=PANEL,
                           outline=CYAN, width=3)
    for row, (text, colour) in enumerate(notes):
        draw.text((x, 300 + row * 75), text, font=_font(29, row == 0), fill=colour)
    _save(image, "slide_04_icp_pose_graph.png")


def make_slide_5(figures: dict[str, Path]) -> None:
    image, draw = _canvas("Rigid result and its remaining branch mismatch",
                          "The marker and trunk align well; most visible disagreement remains near branch ends")
    _panel(image, figures["front"], (90, 180, 1740, 770),
           "Code-rendered front-view comparison", GREEN)
    draw.text((425, 985), "A rigid camera pose cannot represent physical bending inside a branch",
              font=_font(30, True), fill=ORANGE)
    _save(image, "slide_05_rigid_result_and_limitation.png")


def make_slide_6(figures: dict[str, Path]) -> None:
    image, draw = _canvas("Which parts are allowed to bend?",
                          "Mobility combines weak cross-view support, branch-like shape, and low density")
    _panel(image, figures["mobility"], (70, 175, 1260, 820),
           "Computed mobility field", ORANGE)
    draw.rounded_rectangle((1380, 210, 1850, 890), radius=22, fill=PANEL,
                           outline=ORANGE, width=3)
    lines = [
        ("Weak agreement", "< 3 supporting views"),
        ("Thin geometry", "PCA in a 75 mm area"),
        ("Low density", "branch ends, not junctions"),
        ("Forced anchor", "bottom 20% is fixed"),
        ("Control nodes", "spaced about 80 mm"),
    ]
    for row, (heading, detail) in enumerate(lines):
        y = 260 + row * 122
        draw.text((1420, y), heading, font=_font(28, True), fill=ORANGE)
        draw.text((1420, y + 42), detail, font=_font(22), fill=WHITE)
    _save(image, "slide_06_branch_tip_mobility.png")


def make_slide_7(figures: dict[str, Path]) -> None:
    image, draw = _canvas("All-view branch-tip deformation",
                          "Each frame moves toward the median of agreeing views; all frame updates are then applied together")
    _panel(image, figures["movement"], (65, 180, 1160, 800),
           "Estimated displacement", ORANGE)
    _panel(image, figures["side"], (1270, 180, 585, 800),
           "Rigid / non-rigid side view", GREEN)
    draw.text((490, 1000), "Smooth node motion • strong static anchors • maximum tip movement 80 mm",
              font=_font(28, True), fill=WHITE)
    _save(image, "slide_07_nonrigid_consensus_and_movement.png")


def make_slide_8(figures: dict[str, Path]) -> None:
    image, draw = _canvas("Final rigid versus non-rigid fitting",
                          "Both completed clouds are projected into held-out camera views with the same poses")
    _panel(image, figures["projection"], (55, 180, 1210, 810),
           "Example projection comparison", CYAN)
    _panel(image, figures["metrics"], (1310, 180, 555, 810),
           "Global metrics", GREEN)
    draw.text((340, 1010), "Green = match   Red = extra projection   Blue = missing observed surface",
              font=_font(27, True), fill=WHITE)
    _save(image, "slide_08_projection_evaluation.png")


def main() -> None:
    marker = _marker_paths()
    figures = _comparison_paths()
    missing = [path for path in [*marker.values(), *figures.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required figure(s):\n" + "\n".join(map(str, missing)))
    frame_ids, initial, final, marker_center = _load_poses()

    make_slide_1(marker, figures)
    make_slide_2(marker)
    make_slide_3(frame_ids, initial, final, marker_center)
    make_slide_4(frame_ids, final)
    make_slide_5(figures)
    make_slide_6(figures)
    make_slide_7(figures)
    make_slide_8(figures)


if __name__ == "__main__":
    main()
