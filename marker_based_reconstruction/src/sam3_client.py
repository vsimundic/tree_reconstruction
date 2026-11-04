"""HTTP client for the host-side SAM 3 mask-prompt segmentation service.

SAM 3 runs on the HOST (conda ``sam3`` env, GPU) behind ``simundic/sam3_server.py``.
This container is CPU-only with no torch, so we segment by POSTing to that service
over the docker bridge network. Depends only on the stdlib + numpy (both guaranteed
in the container image) -- no ``requests``/torch needed.

Wire contract (confirmed against the live SAM 3 model)::

    POST {base}/segment
      multipart/form-data, field ``file`` = a .npz (np.savez) containing:
        image        (H,W,3) uint8              required  RGB / pseudo-RGB image
        mask         (H,W)   uint8/bool         optional  binary map used as a PROMPT
        points       (K,2)   int  [x,y]         optional
        point_labels (K,)    int  1=fg 0=bg     optional
        box          (4,)    int  [x0,y0,x1,y1] optional
        multimask    scalar  bool               optional (default False)
      At least one of mask / points / box must be present.
    -> 200: a .npz (application/octet-stream) containing:
        masks          (N,H,W) uint8 0/1 at original resolution
        scores         (N,)    float32
        low_res_logits (N,288,288) float32
       4xx/5xx: JSON {"error": "..."}

    GET {base}/health -> {"status","device","checkpoint","mask_input_hw"}

Address resolution: set ``SAM3_URL`` to override; otherwise we try
``host.docker.internal`` (works once run.sh adds ``--add-host=...:host-gateway``)
and fall back to the docker bridge gateway ``172.17.0.1`` (== the host).

Typical use::

    import sam3_client
    out = sam3_client.segment(rgb, mask=binary_map)   # rgb (H,W,3) uint8, binary_map (H,W)
    masks = out["masks"]                              # (N,H,W) uint8

    # depth / intensity / binary source instead of a photo:
    img = sam3_client.to_uint8_rgb(depth)             # normalize + broadcast to (H,W,3)
    out = sam3_client.segment(img, mask=binary_map, multimask=True)

    # point-only prompt, no mask at all:
    out = sam3_client.segment(img, points=[[x, y]], point_labels=[1])
"""
from __future__ import annotations

import io
import json
import os
import socket
import urllib.error
import urllib.request
import uuid

import numpy as np

DEFAULT_PORT = 8000


class Sam3ServerError(RuntimeError):
    """Raised when the SAM 3 service returns an error or is unreachable."""


def _resolve_base_url() -> str:
    """Base URL of the host SAM 3 service. Override with the ``SAM3_URL`` env var."""
    env = os.environ.get("SAM3_URL")
    if env:
        return env.rstrip("/")
    for host in ("host.docker.internal", "172.17.0.1"):
        try:
            socket.gethostbyname(host)
            return f"http://{host}:{DEFAULT_PORT}"
        except OSError:
            continue
    return f"http://172.17.0.1:{DEFAULT_PORT}"


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Coerce a 2-D/3-D array to ``(H,W,3)`` uint8 for the service ``image``
    field, min-max normalizing non-uint8/grayscale input (broadcasting
    single-channel to 3). A uint8 ``(H,W,3)`` array passes through unchanged."""
    a = np.asarray(image)
    if a.ndim == 3 and a.shape[2] == 3 and a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    if a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 3):
        f = a.astype(np.float64)
        lo, hi = float(f.min()), float(f.max())
        norm = np.zeros_like(f) if hi <= lo else (f - lo) / (hi - lo)
        u8 = (norm * 255.0).round().astype(np.uint8)
        if u8.ndim == 2:
            u8 = np.repeat(u8[:, :, None], 3, axis=2)
        return np.ascontiguousarray(u8)
    raise ValueError(f"expected (H,W) or (H,W,3), got shape {a.shape}")


def segment(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    points=None,
    point_labels=None,
    box=None,
    multimask: bool = False,
    base_url: str | None = None,
    timeout: float = 120.0,
) -> dict:
    """Segment ``image`` via the host SAM 3 service, prompted by any combination
    of ``mask``/``points``/``box`` (at least one required). ``image`` must be
    ``(H,W,3)`` uint8 (2-D auto-converted via :func:`to_uint8_rgb`). Returns
    ``{masks, scores, low_res_logits}``; raises :class:`Sam3ServerError` on
    transport or server errors."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = to_uint8_rgb(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must be (H,W,3); got {image.shape}. "
            "Use sam3_client.to_uint8_rgb() for grayscale/depth/intensity input."
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"image must be uint8; got {image.dtype}. "
            "Use sam3_client.to_uint8_rgb() to normalize."
        )
    if mask is None and points is None and box is None:
        raise ValueError("at least one of mask, points, or box is required")

    arrays = {
        "image": np.ascontiguousarray(image, dtype=np.uint8),
        "multimask": np.asarray(bool(multimask)),
    }
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2-D (H,W); got shape {mask.shape}")
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"mask shape {mask.shape} does not match image HxW {image.shape[:2]}"
            )
        arrays["mask"] = (mask > 0).astype(np.uint8)
    if points is not None:
        arrays["points"] = np.asarray(points, dtype=np.int64).reshape(-1, 2)
    if point_labels is not None:
        arrays["point_labels"] = np.asarray(point_labels, dtype=np.int64).reshape(-1)
    if box is not None:
        arrays["box"] = np.asarray(box, dtype=np.int64).reshape(4)

    buf = io.BytesIO()
    np.savez(buf, **arrays)

    base = (base_url or _resolve_base_url()).rstrip("/")
    data = _post_file(f"{base}/segment", buf.getvalue(), timeout=timeout)

    with np.load(io.BytesIO(data)) as npz:
        out = {"masks": npz["masks"], "scores": npz["scores"]}
        if "low_res_logits" in npz.files:
            out["low_res_logits"] = npz["low_res_logits"]
    return out


def health(base_url: str | None = None, timeout: float = 10.0) -> dict:
    """Return the service ``/health`` payload, or raise :class:`Sam3ServerError`."""
    base = (base_url or _resolve_base_url()).rstrip("/")
    url = f"{base}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise Sam3ServerError(
            f"could not reach SAM 3 service at {url}: {getattr(exc, 'reason', exc)}. "
            "Is sam3_server.py running on the host (uvicorn ... --host 0.0.0.0)?"
        ) from exc


def _post_file(url: str, payload: bytes, *, timeout: float, filename: str = "req.npz") -> bytes:
    boundary = "----sam3-" + uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    body = head + payload + tail
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise Sam3ServerError(_error_message(exc)) from exc
    except urllib.error.URLError as exc:
        raise Sam3ServerError(
            f"could not reach SAM 3 service at {url}: {exc.reason}. "
            "Is sam3_server.py running on the host (uvicorn ... --host 0.0.0.0)? "
            "Override the address with the SAM3_URL env var."
        ) from exc


def _error_message(exc: urllib.error.HTTPError) -> str:
    try:
        detail = json.loads(exc.read().decode())
        if isinstance(detail, dict) and "error" in detail:
            return f"SAM 3 service error (HTTP {exc.code}): {detail['error']}"
    except Exception:
        pass
    return f"SAM 3 service error (HTTP {exc.code})"


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=None, help="override base URL (else SAM3_URL / auto)")
    parser.add_argument("--smoke", action="store_true", help="run a synthetic segmentation")
    args = parser.parse_args()

    base = args.url or _resolve_base_url()
    print(f"[sam3_client] service base URL: {base}")
    print(f"[sam3_client] /health: {health(base)}")

    if args.smoke:
        H, W = 360, 480
        rng = np.random.default_rng(0)
        img = rng.integers(0, 60, (H, W, 3), dtype=np.uint8)
        m1 = np.zeros((H, W), np.uint8); m1[80:200, 100:260] = 1
        m2 = np.zeros((H, W), np.uint8); m2[220:320, 300:440] = 1
        a = segment(img, mask=m1, base_url=base)
        b = segment(img, mask=m2, base_url=base)
        print(f"  A: masks {a['masks'].shape} scores {np.asarray(a['scores']).round(3)}")
        print(f"  B: masks {b['masks'].shape} scores {np.asarray(b['scores']).round(3)}")
        pa, pb = a["masks"][0] > 0, b["masks"][0] > 0
        union = int(np.logical_or(pa, pb).sum())
        iou = int(np.logical_and(pa, pb).sum()) / union if union else 0.0
        print(f"  IoU(A,B) = {iou:.3f}  (low => the mask prompt drives the output)")


if __name__ == "__main__":
    _main()
