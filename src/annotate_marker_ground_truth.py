#!/usr/bin/env python3
"""Click marker centers in intensity/depth frames and save them as ground truth."""

import os
from pathlib import Path

import cv2
import numpy as np
import yaml

import config


WINDOW_NAME = "Marker ground-truth annotation"
HEADER_HEIGHT = 42
DEPTH_SCALE = 0.001

# ---------------------------------------------------------------------------
# EDIT THESE VARIABLES
# ---------------------------------------------------------------------------

# Directory containing <id>_intensity.png and <id>_depth.png.
FRAME_DIR = config.CFG.frame_dir

# None writes marker_manual_pos.yaml beside FRAME_DIR. Set a path string to
# write elsewhere.
OUTPUT_PATH = None

# Resize the application window while preserving source-image click coordinates.
DISPLAY_SCALE = 1.0


def discover_frames(frame_dir):
    """Return frame IDs having both an intensity and a depth image."""
    ids = []
    for depth_path in frame_dir.glob("*_depth.png"):
        frame_id = depth_path.name[:-len("_depth.png")]
        if (frame_dir / f"{frame_id}_intensity.png").is_file():
            ids.append(frame_id)
    try:
        return sorted(ids, key=int)
    except ValueError:
        return sorted(ids)


def brighten_intensity(image, gamma=0.5):
    value = image.astype(np.float32)
    value /= value.max() + 1e-6
    return (np.power(value, gamma) * 255).astype(np.uint8)


def colorize_depth(depth_raw):
    depth_m = depth_raw.astype(np.float32) * DEPTH_SCALE
    valid = depth_m > 0
    normalized = np.zeros_like(depth_m)
    if valid.any():
        lo, hi = np.percentile(depth_m[valid], (2, 98))
        normalized[valid] = np.clip(
            (depth_m[valid] - lo) / (hi - lo + 1e-6), 0, 1)
    colored = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def load_annotations(path):
    if not path.exists():
        return {}
    with path.open() as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def save_annotations(path, annotations):
    """Atomically update the YAML so an interrupted write cannot destroy it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        yaml.safe_dump(annotations, stream, sort_keys=False, default_flow_style=None)
    os.replace(temporary, path)


def annotation_key(frame_id):
    return f"frame{frame_id}"


class MarkerAnnotator:
    def __init__(self, frame_dir, output_path, scale=1.0):
        self.frame_dir = frame_dir
        self.output_path = output_path
        self.scale = scale
        self.frame_ids = discover_frames(frame_dir)
        if not self.frame_ids:
            raise FileNotFoundError(
                f"No matching *_intensity.png / *_depth.png frames in {frame_dir}")
        self.annotations = load_annotations(output_path)
        self.index = 0
        self.image_shape = None
        self.status = ""

    @property
    def frame_id(self):
        return self.frame_ids[self.index]

    def set_point(self, point):
        self.annotations[annotation_key(self.frame_id)] = [
            int(point[0]), int(point[1])]
        save_annotations(self.output_path, self.annotations)
        self.status = f"saved ({point[0]}, {point[1]})"

    def delete_point(self):
        removed = self.annotations.pop(annotation_key(self.frame_id), None)
        if removed is not None:
            save_annotations(self.output_path, self.annotations)
            self.status = "annotation deleted"
        else:
            self.status = "frame was not annotated"

    def move(self, offset):
        self.index = (self.index + offset) % len(self.frame_ids)
        self.status = ""

    def render(self):
        intensity = cv2.imread(
            str(self.frame_dir / f"{self.frame_id}_intensity.png"),
            cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(
            str(self.frame_dir / f"{self.frame_id}_depth.png"),
            cv2.IMREAD_UNCHANGED)
        if intensity is None or depth is None:
            raise OSError(f"Could not load frame {self.frame_id}")
        if intensity.ndim == 3:
            intensity = cv2.cvtColor(intensity, cv2.COLOR_BGR2GRAY)

        left = cv2.cvtColor(brighten_intensity(intensity), cv2.COLOR_GRAY2BGR)
        right = colorize_depth(depth)
        if left.shape[:2] != right.shape[:2]:
            raise ValueError(
                f"Intensity/depth shape mismatch in frame {self.frame_id}: "
                f"{left.shape[:2]} != {right.shape[:2]}")
        self.image_shape = left.shape[:2]

        point = self.annotations.get(annotation_key(self.frame_id))
        if point is not None:
            marker_point = (int(point[0]), int(point[1]))
            for panel in (left, right):
                cv2.drawMarker(
                    panel, marker_point, (255, 0, 255),
                    cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
                cv2.circle(panel, marker_point, 5, (255, 255, 255), 1, cv2.LINE_AA)

        body = np.hstack((left, right))
        header = np.zeros((HEADER_HEIGHT, body.shape[1], 3), np.uint8)
        annotated_count = sum(
            annotation_key(frame_id) in self.annotations
            for frame_id in self.frame_ids)
        title = (
            f"Frame {self.frame_id} ({self.index + 1}/{len(self.frame_ids)}) | "
            f"annotated {annotated_count}/{len(self.frame_ids)} | "
            "click either image | A/D or arrows: frame | X: delete | Q: quit"
        )
        cv2.putText(
            header, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (230, 230, 230), 1, cv2.LINE_AA)
        detail = f"Intensity (gamma display only)             Depth (display only)"
        if self.status:
            detail += f"             {self.status}"
        cv2.putText(
            header, detail, (8, 36), cv2.FONT_HERSHEY_SIMPLEX,
            0.43, (80, 220, 120), 1, cv2.LINE_AA)
        display = np.vstack((header, body))
        if self.scale != 1.0:
            display = cv2.resize(
                display, None, fx=self.scale, fy=self.scale,
                interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_NEAREST)
        return display

    def mouse_callback(self, event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN or self.image_shape is None:
            return
        display_x = int(round(x / self.scale))
        display_y = int(round(y / self.scale)) - HEADER_HEIGHT
        height, width = self.image_shape
        if not (0 <= display_y < height and 0 <= display_x < 2 * width):
            return
        self.set_point((display_x % width, display_y))

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)
        print(f"Annotations are saved immediately to {self.output_path}")
        try:
            while True:
                cv2.imshow(WINDOW_NAME, self.render())
                key = cv2.waitKeyEx(30)
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("d"), ord("D"), ord("n"), ord("N"), 2555904, 65363):
                    self.move(1)
                elif key in (ord("a"), ord("A"), ord("p"), ord("P"), 2424832, 65361):
                    self.move(-1)
                elif key in (ord("x"), ord("X"), 8, 127):
                    self.delete_point()
        finally:
            cv2.destroyAllWindows()


def main():
    if DISPLAY_SCALE <= 0:
        raise ValueError("DISPLAY_SCALE must be greater than zero")
    frame_dir = Path(FRAME_DIR).expanduser().resolve()
    output_path = (
        Path(OUTPUT_PATH).expanduser().resolve()
        if OUTPUT_PATH else frame_dir.parent / "marker_manual_pos.yaml"
    )
    MarkerAnnotator(frame_dir, output_path, DISPLAY_SCALE).run()


if __name__ == "__main__":
    main()
