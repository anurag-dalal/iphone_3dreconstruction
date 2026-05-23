"""
datareader.py — loads a single frame (RGB, depth, confidence) from an ARKit dataset folder.
"""

import json
import os
import numpy as np
import cv2
from PIL import Image
import pillow_heif


pillow_heif.register_heif_opener()


def load_frame(frame_dir: str, frame_id: str, proc_h: int, proc_w: int):
    """
    Load and resize one frame.

    All outputs are in **sensor-native landscape** orientation to match the
    intrinsics calibration space.  The depth map is stored as (W=192, H=144)
    in landscape; we always process at that native landscape resolution so that
    intrinsic scaling is uniform (scale = depth_dim / reference_dim = 0.1).

    proc_h / proc_w from the config represent the two processing dimensions
    (192 and 144). We choose whichever layout matches the stored depth
    so that dW = max(proc_h, proc_w) and dH = min(proc_h, proc_w).

    Returns
    -------
    dict with keys:
        rgb        : np.uint8  (dH, dW, 3)  — landscape sensor-native
        depth      : np.float32 (dH, dW)   — metres, NaN where invalid
        confidence : np.uint8   (dH, dW)   — 0/1/2
        meta       : dict parsed from JSON
    """
    base = os.path.join(frame_dir, f"frame_{frame_id}")

    # ── JSON metadata ──────────────────────────────────────────────────────────
    with open(f"{base}.json") as f:
        meta = json.load(f)

    depth_res = meta["depthResolutionStored"]   # {"width": 192, "height": 144}
    dW, dH = depth_res["width"], depth_res["height"]  # landscape: W > H

    # ── Depth (uint16 LE, millimetres) ─────────────────────────────────────────
    # reshape(dH, dW) = reshape(144, 192) — sensor-native landscape rows×cols
    depth_raw = np.fromfile(f"{base}.depth_u16.bin", dtype="<u2").reshape(dH, dW)
    depth_m = depth_raw.astype(np.float32) * 0.001
    depth_m[depth_raw == 0] = np.nan          # invalid sentinel → NaN

    # ── Confidence (uint8, 0/1/2) ──────────────────────────────────────────────
    conf = np.fromfile(f"{base}.conf_u8.bin", dtype=np.uint8).reshape(dH, dW)

    # ── RGB (HEIC, sensor-native landscape) ────────────────────────────────────
    rgb_img = Image.open(f"{base}.rgb.heic").convert("RGB")
    rgb = np.array(rgb_img)                   # (720, 960, 3) uint8 landscape

    # ── Resize RGB to depth resolution (landscape) ─────────────────────────────
    # cv2.resize takes (dst_cols, dst_rows) = (dW, dH)
    rgb_resized = cv2.resize(rgb, (dW, dH), interpolation=cv2.INTER_LINEAR)

    return {
        "rgb":        rgb_resized,   # (dH, dW, 3)
        "depth":      depth_m,       # (dH, dW)
        "confidence": conf,          # (dH, dW)
        "meta":       meta,
    }



def iter_frames(frame_dir: str, proc_h: int, proc_w: int):
    """
    Generator that yields frames in capture order.
    Skips frames where any file is missing or tracking is unreliable.
    proc_h / proc_w are passed for API compatibility but the actual
    processing resolution is determined from the stored depth dimensions.
    """
    json_files = sorted(
        f for f in os.listdir(frame_dir) if f.endswith(".json")
    )
    for jf in json_files:
        frame_id = jf[len("frame_"):len("frame_000000")]  # 6-char id
        base = os.path.join(frame_dir, f"frame_{frame_id}")
        if not all(os.path.exists(f"{base}{ext}") for ext in
                   (".json", ".depth_u16.bin", ".conf_u8.bin", ".rgb.heic")):
            continue
        try:
            frame = load_frame(frame_dir, frame_id, proc_h, proc_w)
            # Skip frames with unreliable ARKit pose
            if frame["meta"].get("trackingState", "normal") != "normal":
                continue
            yield frame_id, frame
        except Exception as e:
            print(f"[datareader] Skipping frame {frame_id}: {e}")
