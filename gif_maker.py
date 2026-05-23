"""
gif_maker.py — 360° rotating GIF from a PLY file (point cloud or mesh)
Usage:
    python gif_maker.py <input.ply> <output.gif> [options]

Options:
    --frames N          Number of frames for a full 360° rotation  (default: 60)
    --fps N             GIF playback speed in frames-per-second    (default: 20)
    --width W           Render width in pixels                     (default: 800)
    --height H          Render height in pixels                     (default: 600)
    --axis {x,y,z}      Rotation axis                              (default: y)
    --bg R G B          Background colour 0-255 each               (default: 25 25 25)
    --point-size S      Point size for point clouds                (default: 2.0)
    --elevation DEG     Camera elevation angle above the object    (default: 20)

Examples:
    python gif_maker.py output/foo/downsampled_pointcloud.ply spin.gif
    python gif_maker.py output/foo/architectural_elements_mesh.ply mesh.gif --frames 72 --fps 24
"""

import argparse
import os
import sys
import io
import numpy as np
import open3d as o3d
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    """Return a 3×3 rotation matrix around 'axis' ('x'|'y'|'z')."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == "y":
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]], dtype=np.float64)
    if axis == "x":
        return np.array([[1,  0,  0],
                         [0,  c, -s],
                         [0,  s,  c]], dtype=np.float64)
    # z
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], dtype=np.float64)


def _place_camera(vis: o3d.visualization.Visualizer,
                  centroid: np.ndarray,
                  radius: float,
                  heading_rad: float,
                  elevation_deg: float,
                  width: int,
                  height: int) -> None:
    """Point the Open3D camera at 'centroid' from a spherical position."""
    el  = np.radians(elevation_deg)
    eye = centroid + radius * np.array([
        np.cos(el) * np.sin(heading_rad),
        np.sin(el),
        np.cos(el) * np.cos(heading_rad),
    ])
    up = np.array([0.0, 1.0, 0.0])
    # If the camera is nearly vertical, switch up-vector
    fwd = centroid - eye
    fwd /= np.linalg.norm(fwd)
    if abs(np.dot(fwd, up)) > 0.95:
        up = np.array([0.0, 0.0, 1.0])

    ctr = vis.get_view_control()
    ctr.set_lookat(centroid.tolist())
    ctr.set_front((-fwd).tolist())
    ctr.set_up(up.tolist())
    ctr.set_zoom(0.55)


def _geometry_info(geom):
    """Return (centroid, diagonal_radius) for either a PointCloud or TriangleMesh."""
    if isinstance(geom, o3d.geometry.PointCloud):
        pts = np.asarray(geom.points)
    else:
        pts = np.asarray(geom.vertices)
    centroid = pts.mean(axis=0)
    radius   = float(np.linalg.norm(pts - centroid, axis=1).max())
    return centroid, max(radius, 1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# Core render loop
# ──────────────────────────────────────────────────────────────────────────────

def make_gif(
    input_path:    str,
    output_path:   str,
    n_frames:      int   = 60,
    fps:           int   = 20,
    width:         int   = 800,
    height:        int   = 600,
    axis:          str   = "y",
    bg_color:      tuple = (25, 25, 25),
    point_size:    float = 2.0,
    elevation_deg: float = 20.0,
) -> None:
    print(f"Loading  : {input_path}")

    # ── Detect file type and load ────────────────────────────────────────────
    # Try mesh first (a mesh PLY is also readable as a point cloud, so check
    # triangles explicitly before falling back to point-cloud mode).
    is_mesh = False
    _mesh_probe = o3d.io.read_triangle_mesh(input_path)
    if len(_mesh_probe.triangles) > 0:
        geom    = _mesh_probe
        is_mesh = True
        geom.compute_vertex_normals()
        print(f"Type     : mesh  ({len(geom.vertices):,} verts, "
              f"{len(geom.triangles):,} tris)")
    else:
        geom = o3d.io.read_point_cloud(input_path)
        if len(geom.points) == 0:
            sys.exit(f"[error] Could not load any geometry from {input_path}")
        print(f"Type     : point cloud  ({len(geom.points):,} pts)")

    centroid, radius = _geometry_info(geom)
    print(f"Centroid : {centroid.round(3)}")
    print(f"Radius   : {radius:.3f} m")
    print(f"Rendering: {n_frames} frames  {width}×{height}  axis={axis}  "
          f"elevation={elevation_deg}°")

    # ── Open3D off-screen window ─────────────────────────────────────────────
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)

    ropt = vis.get_render_option()
    ropt.background_color = np.array([c / 255.0 for c in bg_color])
    if not is_mesh:
        ropt.point_size = point_size
    ropt.mesh_show_back_face   = True
    ropt.light_on              = True

    vis.add_geometry(geom)

    angles  = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    frames  = []

    for i, angle in enumerate(angles):
        _place_camera(vis, centroid, radius * 2.8, angle, elevation_deg, width, height)
        vis.poll_events()
        vis.update_renderer()

        # Capture frame as PIL Image
        raw = vis.capture_screen_float_buffer(do_render=True)
        img_np = (np.asarray(raw) * 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np))

        if (i + 1) % 10 == 0 or i == n_frames - 1:
            print(f"  frame {i+1:3d}/{n_frames}", end="\r", flush=True)

    print()
    vis.destroy_window()

    # ── Save GIF ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    duration_ms = max(1, round(1000 / fps))

    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved    : {output_path}  ({size_kb:.0f} KB,  "
          f"{n_frames} frames @ {fps} fps)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Make a 360° rotating GIF from a PLY point cloud or mesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input",  help="Input PLY file (point cloud or mesh)")
    p.add_argument("output", help="Output GIF file")
    p.add_argument("--frames",     type=int,   default=60,
                   help="Number of frames (default: 60)")
    p.add_argument("--fps",        type=int,   default=20,
                   help="Frames per second (default: 20)")
    p.add_argument("--width",      type=int,   default=800,
                   help="Render width in pixels (default: 800)")
    p.add_argument("--height",     type=int,   default=600,
                   help="Render height in pixels (default: 600)")
    p.add_argument("--axis",       choices=["x", "y", "z"], default="y",
                   help="Rotation axis (default: y)")
    p.add_argument("--bg",         nargs=3, type=int,
                   metavar=("R", "G", "B"), default=[25, 25, 25],
                   help="Background colour 0-255 (default: 25 25 25)")
    p.add_argument("--point-size", type=float, default=2.0,
                   help="Point size for point clouds (default: 2.0)")
    p.add_argument("--elevation",  type=float, default=20.0,
                   help="Camera elevation above the object in degrees (default: 20)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_gif(
        input_path    = args.input,
        output_path   = args.output,
        n_frames      = args.frames,
        fps           = args.fps,
        width         = args.width,
        height        = args.height,
        axis          = args.axis,
        bg_color      = tuple(args.bg),
        point_size    = args.point_size,
        elevation_deg = args.elevation,
    )
