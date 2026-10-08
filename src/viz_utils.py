"""
viz_utils.py
============
Small visualization helpers shared by the localization scripts:
- 2D montage building (titled, resized tiles in a grid)
- intensity brightening / colormaps
- 3D helpers (bounding boxes, coordinate frame, plane-normal arrow, color-by-value)
"""

import cv2
import numpy as np
import open3d as o3d


# --------------------------------------------------------------------------
# 2D
# --------------------------------------------------------------------------

def brighten(intensity, gamma=0.4):
    """uint16/float intensity -> 8-bit, gamma-stretched so the dark ToF image is
    actually visible."""
    g = intensity.astype(np.float32)
    g = g / (g.max() + 1e-6)
    g = np.power(g, gamma)
    return (g * 255).astype(np.uint8)


def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def colorize(value_img, cmap=cv2.COLORMAP_JET, valid=None):
    """Normalize a float image to 0-255 and apply a colormap. `valid` masks the
    pixels used for normalization (e.g. depth > 0)."""
    v = value_img.astype(np.float32)
    m = valid if valid is not None else np.ones_like(v, bool)
    if m.any():
        lo, hi = np.percentile(v[m], 2), np.percentile(v[m], 98)
    else:
        lo, hi = v.min(), v.max()
    v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    out = cv2.applyColorMap((v * 255).astype(np.uint8), cmap)
    if valid is not None:
        out[~valid] = 0
    return out


def color_labels(labels):
    """Colorize a connected-components label image (0 = background → black)."""
    n = int(labels.max()) + 1
    rng = np.random.default_rng(0)
    colors = rng.integers(50, 255, size=(max(n, 1), 3), dtype=np.uint8)
    if n > 0:
        colors[0] = (0, 0, 0)
    return colors[labels]


def put_title(bgr, text, color=(40, 220, 40)):
    out = bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA)
    return out


def montage(panels, cols=3, tile_w=420):
    """panels: list of (title, bgr_image). Returns a single tiled BGR image."""
    tiles = []
    for title, img in panels:
        img = to_bgr(img)
        h, w = img.shape[:2]
        tw = tile_w
        th = int(h * tw / w)
        img = cv2.resize(img, (tw, th))
        tiles.append(put_title(img, title))
    if not tiles:
        return np.zeros((10, 10, 3), np.uint8)

    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def show_scrollable(window_name, img, viewport_h=800):
    """Show a tall image (e.g. a multi-row montage) at FULL resolution -- so every
    tile stays readable -- with a vertical scrollbar (trackbar) to reach rows that
    don't fit on screen at once. Blocks until Enter/Esc/q, mirroring the previous
    `cv2.waitKey(0)` behaviour. If `img` already fits within `viewport_h`, this is
    just a plain imshow (no trackbar)."""
    h, w = img.shape[:2]
    viewport_h = min(viewport_h, h)
    max_scroll = h - viewport_h

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w, viewport_h)
    if max_scroll > 0:
        cv2.createTrackbar("scroll", window_name, 0, max_scroll, lambda pos: None)

    while True:
        pos = cv2.getTrackbarPos("scroll", window_name) if max_scroll > 0 else 0
        cv2.imshow(window_name, img[pos:pos + viewport_h])
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 27, ord('q')):  # Enter, Esc, or q closes
            break
    cv2.destroyWindow(window_name)


# --------------------------------------------------------------------------
# 3D
# --------------------------------------------------------------------------

def aabb(pcd, color=(1, 1, 0)):
    box = pcd.get_axis_aligned_bounding_box()
    box.color = color
    return box


def obb(pcd, color=(1, 0.6, 0)):
    box = pcd.get_oriented_bounding_box()
    box.color = color
    return box


def coord_frame(size=0.1, origin=(0, 0, 0)):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)


def color_by_value(values, cmap=cv2.COLORMAP_VIRIDIS):
    """Map an (N,) array to (N,3) RGB floats in [0,1] via an OpenCV colormap."""
    v = values.astype(np.float32)
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    bgr = cv2.applyColorMap((v * 255).astype(np.uint8).reshape(-1, 1), cmap)
    rgb = bgr[:, 0, ::-1].astype(np.float32) / 255.0
    return rgb


def distinct_colors(n):
    """`n` visually distinct RGB colors (n,3 floats in [0,1]). Hues are stepped
    by the golden-ratio conjugate so any count spreads evenly around the wheel
    without a fixed palette to run out of -- handy for coloring each frame's
    cloud its own color in a merged reconstruction."""
    idx = np.arange(n, dtype=np.float32)
    hue = np.mod(idx * 0.6180339887, 1.0)            # golden-ratio hue stepping
    hsv = np.stack([hue * 179,                        # OpenCV hue is 0-179
                    np.full(n, 200, np.float32),      # strong but not neon
                    np.full(n, 245, np.float32)], axis=1)
    bgr = cv2.cvtColor(hsv[None].astype(np.uint8), cv2.COLOR_HSV2BGR)[0]
    return bgr[:, ::-1].astype(np.float32) / 255.0


def normal_arrow(center, normal, length=0.15, color=(0.1, 0.4, 1.0)):
    """A simple line from `center` along `normal` (unit or not) of given length."""
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-9)
    pts = np.array([center, np.asarray(center) + n * length])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector([[0, 1]]))
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls
