"""
backproject.py — depth → coloured 3-D point cloud using ARKit intrinsics/extrinsics.
"""

import numpy as np


def get_intrinsics(meta: dict, actual_h: int, actual_w: int) -> tuple:
    """
    Parse the camera intrinsic matrix and scale it from the reference resolution
    to the actual array resolution (sensor-native landscape).

    Both reference and actual are in landscape, so the scale is uniform:
      scale = actual_w / ref_w = actual_h / ref_h  (e.g. 192/1920 = 0.1)

    Returns (fx, fy, cx, cy) in actual-resolution pixel units.
    """
    vals = meta["intrinsics"]["values"]          # 9 floats, column-major
    # column-major layout: [fx, 0, 0,  0, fy, 0,  cx, cy, 1]
    fx_ref = vals[0]
    fy_ref = vals[4]
    cx_ref = vals[6]
    cy_ref = vals[7]

    ref_res = meta["intrinsicsReferenceResolution"]
    ref_w = ref_res["width"]    # e.g. 1920
    ref_h = ref_res["height"]   # e.g. 1440

    # Scale uniformly from reference landscape to actual landscape array dims
    sx = actual_w / ref_w      # e.g. 192/1920 = 0.1
    sy = actual_h / ref_h      # e.g. 144/1440 = 0.1  (should equal sx)

    return fx_ref * sx, fy_ref * sy, cx_ref * sx, cy_ref * sy


def get_world_from_camera(meta: dict) -> np.ndarray:
    """Return the 4×4 worldFromCamera matrix (column-major → row-major)."""
    vals = meta["worldFromCamera"]["values"]     # 16 floats, column-major
    return np.array(vals, dtype=np.float64).reshape(4, 4, order="F")


def backproject_frame(frame: dict, proc_h: int, proc_w: int,
                      min_confidence: int = 1, max_depth: float = 5.0):
    """
    Back-project a single frame's depth map into world-space 3-D points.

    Coordinate conventions (ARKit / OpenGL):
      Camera frame: +X right, +Y up, -Z forward.
      Image frame:  u (col) increases right (+X), v (row) increases DOWN (-Y).
      ARKit depth d is the perpendicular (z-axis) distance → z_cam = -d.
      Because image +v is opposite to camera +Y:
          y_cam = -(v - cy) / fy * d      (note the minus sign)

    Returns
    -------
    points : (N, 3) float32  — world-space XYZ
    colors : (N, 3) float32  — RGB in [0, 1]
    """
    depth  = frame["depth"]        # (H, W) float32, NaN = invalid
    conf   = frame["confidence"]   # (H, W) uint8
    rgb    = frame["rgb"]          # (H, W, 3) uint8
    meta   = frame["meta"]

    # Use actual array dimensions — always sensor-native landscape
    H, W = depth.shape             # e.g. (144, 192)
    fx, fy, cx, cy = get_intrinsics(meta, H, W)
    T_wc = get_world_from_camera(meta)

    # ── Build pixel grid ───────────────────────────────────────────────────────
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # ── Validity mask ──────────────────────────────────────────────────────────
    valid = (
        np.isfinite(depth) &
        (depth > 0) &
        (depth <= max_depth) &
        (conf >= min_confidence)
    )

    d  = depth[valid]
    px = xs[valid].astype(np.float32)
    py = ys[valid].astype(np.float32)

    # ── Camera-space points (OpenGL: +X right, +Y up, -Z forward) ─────────────
    # x_cam =  (u - cx) / fx * d        image +u aligns with camera +X ✓
    # y_cam = -(v - cy) / fy * d        image +v goes DOWN, camera +Y goes UP
    # z_cam = -d                         depth is positive, camera -Z forward
    x_cam =  (px - cx) / fx * d
    y_cam = -(py - cy) / fy * d
    z_cam = -d

    ones = np.ones_like(d)
    pts_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=1)  # (N, 4)

    # ── World-space transform ──────────────────────────────────────────────────
    pts_world = (T_wc @ pts_cam.T).T[:, :3].astype(np.float32)  # (N, 3)

    # ── Colour ─────────────────────────────────────────────────────────────────
    colors = rgb[valid].astype(np.float32) / 255.0              # (N, 3)

    return pts_world, colors


def backproject_all_frames(frames_iter, proc_h: int, proc_w: int,
                           min_confidence: int = 1, max_depth: float = 5.0,
                           verbose: bool = True):
    """
    Fuse back-projections from all frames.
    proc_h / proc_w are passed through for API compatibility; actual resolution
    is taken from each frame's array shape (sensor-native landscape).

    Parameters
    ----------
    frames_iter : iterable of (frame_id, frame_dict) from datareader.iter_frames
    Returns (N, 3) points and (N, 3) colors arrays.
    """
    all_pts, all_col = [], []
    for frame_id, frame in frames_iter:
        pts, col = backproject_frame(frame, proc_h, proc_w,
                                     min_confidence, max_depth)
        all_pts.append(pts)
        all_col.append(col)
        if verbose:
            print(f"  frame {frame_id}: {pts.shape[0]:>7,} points")

    if not all_pts:
        raise RuntimeError("No valid points found in any frame.")

    return np.concatenate(all_pts, axis=0), np.concatenate(all_col, axis=0)
