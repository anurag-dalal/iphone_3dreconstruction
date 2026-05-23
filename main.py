"""
main.py — 3-D reconstruction pipeline
Usage: python main.py <config.json>
"""

import sys
import json
import os
import time
import datetime
import numpy as np
import open3d as o3d
import datareader
import backproject as bp
import gif_maker


# ──────────────────────────────────────────────────────────────────────────────
# Logger — tees stdout to a .txt file in the output directory
# ──────────────────────────────────────────────────────────────────────────────

class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath: str):
        self._file = open(filepath, "w", buffering=1)
        self._stdout = sys.__stdout__

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def _init_logger(output_dir: str) -> str:
    log_path = os.path.join(output_dir, "run_log.txt")
    sys.stdout = _Tee(log_path)
    return log_path




# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def numpy_to_o3d(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def _save_gif(ply_path: str, cfg: dict) -> None:
    """If cfg['save_gifs'] is true, render a 360° GIF next to the PLY file."""
    if not cfg.get("save_gifs", False):
        return
    g = cfg.get("gif", {})
    gif_path = os.path.splitext(ply_path)[0] + "_360.gif"
    print(f"  → rendering GIF: {os.path.basename(gif_path)} …")
    gif_maker.make_gif(
        input_path    = ply_path,
        output_path   = gif_path,
        n_frames      = g.get("frames",    48),
        fps           = g.get("fps",       20),
        width         = g.get("width",    640),
        height        = g.get("height",   480),
        axis          = g.get("axis",     "y"),
        bg_color      = tuple(g.get("bg", [25, 25, 25])),
        point_size    = g.get("point_size", 2.0),
        elevation_deg = g.get("elevation",  20.0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Back-project → fused point cloud
# ──────────────────────────────────────────────────────────────────────────────

def stage1_backproject(cfg: dict, proc_h: int, proc_w: int) -> o3d.geometry.PointCloud:
    bp_cfg  = cfg["backproject"]
    dataset = cfg["dataset_path"]

    print("\n[Stage 1] Back-projecting frames …")
    frames = datareader.iter_frames(dataset, proc_h, proc_w)
    points, colors = bp.backproject_all_frames(
        frames, proc_h, proc_w,
        min_confidence=bp_cfg["min_confidence"],
        max_depth=bp_cfg["max_depth"],
    )
    print(f"  total points (raw): {len(points):,}")

    pcd = numpy_to_o3d(points, colors)

    out_path = bp_cfg["output_path"]
    out_path = os.path.join(cfg["output_dir"], os.path.basename(out_path))
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"  saved raw point cloud → {out_path}")
    _save_gif(out_path, cfg)
    cfg["_raw_pcd_path"] = out_path   # stash for Stage 3 colorization
    return pcd


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Downsample / filter
# ──────────────────────────────────────────────────────────────────────────────

def stage2_downsample(pcd: o3d.geometry.PointCloud, cfg: dict) -> o3d.geometry.PointCloud:
    ds_cfg = cfg["pipeline"]["downsample"]
    print("\n[Stage 2] Downsampling / filtering …")

    # Voxel downsampling
    vox = ds_cfg.get("voxel_downsample", {})
    if vox.get("enabled", False):
        pcd = pcd.voxel_down_sample(vox["voxel_size"])
        print(f"  after voxel ({vox['voxel_size']} m): {len(pcd.points):,} pts")

    # Statistical outlier removal
    sor = ds_cfg.get("statistical_outlier_removal", {})
    if sor.get("enabled", False):
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=sor["nb_neighbors"],
            std_ratio=sor["std_ratio"],
        )
        print(f"  after SOR: {len(pcd.points):,} pts")

    # Pass-through filter (axis-aligned bounding box crop)
    pt = ds_cfg.get("pass_through_filter", {})
    if pt.get("enabled", False):
        axis = pt["axis"]          # "x", "y", or "z"
        lo, hi = pt["min_limit"], pt["max_limit"]
        pts = np.asarray(pcd.points)
        col = np.asarray(pcd.colors)
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        mask = (pts[:, axis_idx] >= lo) & (pts[:, axis_idx] <= hi)
        pcd = numpy_to_o3d(pts[mask], col[mask])
        print(f"  after pass-through ({axis} [{lo},{hi}]): {len(pcd.points):,} pts")

    # Overwrite the raw PLY with the filtered version
    out_path = cfg["pipeline"]["downsample"]["output_path"]
    out_path = cfg["output_dir"] + "/" + os.path.basename(out_path)
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"  saved filtered point cloud → {out_path}")
    _save_gif(out_path, cfg)
    return pcd


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: Surface reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def stage3_reconstruct(pcd: o3d.geometry.PointCloud, cfg: dict):
    sr_cfg = cfg["pipeline"]["surface_reconstruction"]
    print("\n[Stage 3] Surface reconstruction …")

    # Estimate and orient normals (required by Poisson and Ball-pivoting)
    print("  estimating normals …")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(100)

    mesh = None
    out_stem = os.path.join(
        cfg["output_dir"],
        os.path.splitext(os.path.basename(
            cfg["pipeline"]["surface_reconstruction"]["output_path"]
        ))[0]
    )

    # ── Poisson ──────────────────────────────────────────────────────────────
    poisson = sr_cfg.get("poisson_reconstruction", {})
    if poisson.get("enabled", False):
        print(f"  Poisson reconstruction (depth={poisson['depth']}) …")
        result = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson["depth"]
        )
        if result is None or (isinstance(result, tuple) and result[0] is None):
            print("  [warn] Poisson returned no mesh — too few points?")
        else:
            mesh, densities = result
            if mesh is None or len(mesh.vertices) == 0:
                print("  [warn] Poisson mesh is empty.")
            else:
                # Trim low-density vertices (artefacts at boundaries)
                dens = np.asarray(densities)
                threshold = np.quantile(dens, 0.05)
                trimmed = mesh.remove_vertices_by_mask(dens < threshold)
                if trimmed is not None and len(trimmed.vertices) > 0:
                    mesh = trimmed
                out_path = f"{out_stem}_poisson.ply"
                o3d.io.write_triangle_mesh(out_path, mesh)
                print(f"  saved Poisson mesh → {out_path}")
                _save_gif(out_path, cfg)

    # ── Ball-pivoting ─────────────────────────────────────────────────────────
    bp_cfg = sr_cfg.get("ball_pivoting", {})
    if bp_cfg.get("enabled", False):
        radii = [float(r) for r in bp_cfg["radii"]]
        print(f"  Ball-pivoting (radii={radii}) …")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        out_path = f"{out_stem}_ballpivot.ply"
        out_path = cfg["output_dir"] + "/" + os.path.basename(out_path)
        o3d.io.write_triangle_mesh(out_path, mesh)
        print(f"  saved Ball-pivoting mesh → {out_path}")
        _save_gif(out_path, cfg)

    # ── Alpha shapes ──────────────────────────────────────────────────────────
    alpha_cfg = sr_cfg.get("alpha_shapes", {})
    if alpha_cfg.get("enabled", False):
        alpha = float(alpha_cfg["alpha"])
        print(f"  Alpha-shape (alpha={alpha}) …")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha
        )
        out_path = f"{out_stem}_alpha.ply"
        out_path = cfg["output_dir"] + "/" + os.path.basename(out_path)
        o3d.io.write_triangle_mesh(out_path, mesh)
        print(f"  saved Alpha-shape mesh → {out_path}")
        _save_gif(out_path, cfg)

    return mesh




def _elapsed(seconds: float) -> str:
    return f"{seconds:.1f}s"

# ──────────────────────────────────────────────────────────────────────────────
# Stage4: Task 1 — Floor plan extraction from PCD
# ──────────────────────────────────────────────────────────────────────────────
def find_floor_plan(pcd: o3d.geometry.PointCloud, cfg: dict):
    """
    Extract a 2-D vector floor plan from the point cloud.

    Furniture is discarded via a per-cell vertical-extent filter:
    walls stretch floor-to-ceiling so each 2-D cell accumulates points
    across many heights (large vertical span).  Furniture surfaces are
    nearly flat and occupy only a narrow height band (small span) and
    are therefore rejected.  Surviving small blobs are removed by a
    connected-component area threshold.

    Steps:
      1. RANSAC floor-plane detection
      2. Orthonormal 2-D basis on the plane
      3. Project all valid points; compute per-cell vertical span
      4. Threshold on vertical span → draft occupancy grid
      5. Remove small connected components (furniture remnants)
      6. Morphological closing to bridge wall gaps
      7. Render & export SVG
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation, binary_closing, label as nd_label

    fp_cfg   = cfg["floor_plan"]
    out_path = os.path.join(cfg["output_dir"],
                            os.path.basename(fp_cfg["output_path"]))
    res             = fp_cfg.get("resolution",           0.05)  # m / cell
    height_lo       = fp_cfg.get("wall_height_min",       0.1)  # m above floor
    height_hi       = fp_cfg.get("wall_height_max",       3.0)  # m above floor
    min_vert_ext    = fp_cfg.get("min_vertical_extent",   0.8)  # m  — furniture filter
    min_comp_area   = fp_cfg.get("min_component_area",    0.5)  # m² — blob filter
    closing_iters   = fp_cfg.get("closing_iterations",    8)    # morphological closing

    points = np.asarray(pcd.points)   # (N, 3) float64

    # ── 1. RANSAC floor plane ────────────────────────────────────────────────
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.02, ransac_n=3, num_iterations=1000
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if normal[1] < 0:   # ensure normal points up (ARKit Y-up world)
        normal = -normal
        d = -d
    print(f"  floor plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    print(f"  floor normal: {normal.round(3)}")

    floor_pts = points[np.asarray(inliers)]
    floor_y   = float(np.median(floor_pts[:, 1]))
    print(f"  floor median Y = {floor_y:.3f} m   inliers: {len(inliers):,}")

    # ── 2. Orthonormal basis on the floor plane ──────────────────────────────
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, ref)) > 0.95:
        ref = np.array([0.0, 0.0, 1.0])
    u = np.cross(normal, ref);  u /= np.linalg.norm(u)   # "right"
    v = np.cross(normal, u);    v /= np.linalg.norm(v)   # "forward"

    # ── 3. Project all height-valid points ──────────────────────────────────
    heights   = points @ normal + d          # signed height above floor
    band_mask = (heights > height_lo) & (heights < height_hi)
    band_pts  = points[band_mask]
    band_h    = heights[band_mask]
    print(f"  points in height band ({height_lo}–{height_hi} m): {len(band_pts):,}")

    if len(band_pts) < 200:
        print("  [warn] too few points in height band — floor plan may be empty")

    x2d = band_pts @ u
    y2d = band_pts @ v

    margin = 0.5
    x_min, x_max = x2d.min() - margin, x2d.max() + margin
    y_min, y_max = y2d.min() - margin, y2d.max() + margin

    nx = max(int((x_max - x_min) / res) + 1, 2)
    ny = max(int((y_max - y_min) / res) + 1, 2)
    print(f"  occupancy grid: {ny}×{nx} cells at {res*100:.0f} cm/cell")

    xi = np.clip(((x2d - x_min) / res).astype(int), 0, nx - 1)
    yi = np.clip(((y2d - y_min) / res).astype(int), 0, ny - 1)

    # Per-cell min/max height → vertical span
    cell_min_h = np.full((ny, nx),  np.inf)
    cell_max_h = np.full((ny, nx), -np.inf)
    np.minimum.at(cell_min_h, (yi, xi), band_h)
    np.maximum.at(cell_max_h, (yi, xi), band_h)
    vert_span = np.where(np.isfinite(cell_min_h), cell_max_h - cell_min_h, 0.0)

    # ── 4. Threshold on vertical span ───────────────────────────────────────
    # Walls: large span.  Furniture surfaces: small span → discarded.
    grid = vert_span >= min_vert_ext
    print(f"  vertical-extent filter (>= {min_vert_ext} m): "
          f"{grid.sum():,} occupied cells (furniture surfaces removed)")

    # ── 5. Remove small connected components (furniture remnants) ───────────
    min_cells = max(1, int(min_comp_area / (res * res)))
    labeled, n_raw = nd_label(grid)
    for comp_id in range(1, n_raw + 1):
        if (labeled == comp_id).sum() < min_cells:
            grid[labeled == comp_id] = False
    n_kept = nd_label(grid)[1]
    print(f"  component filter (min {min_comp_area} m²): "
          f"{n_raw} → {n_kept} regions")

    # ── 6. Morphological closing (bridge wall gaps) + thin dilation ──────────
    grid = binary_closing(grid, iterations=closing_iters)
    grid = binary_dilation(grid, iterations=1)

    # ── 7. Render & save SVG ─────────────────────────────────────────────────
    aspect  = ny / nx
    fig_w   = 14
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * aspect))
    fig.patch.set_facecolor("#f7f4f0")
    ax.set_facecolor("#f7f4f0")

    ax.imshow(
        grid.astype(float), origin="lower", cmap="Greys",
        extent=[x_min, x_max, y_min, y_max],
        aspect="equal", vmin=0, vmax=1, interpolation="nearest",
    )

    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)
    ax.contour(
        x_coords, y_coords, grid.astype(float),
        levels=[0.5], colors=["#1a1a1a"], linewidths=1.8,
    )

    # Scale bar (1 m)
    sb_x = x_min + 0.1 * (x_max - x_min)
    sb_y = y_min + 0.05 * (y_max - y_min)
    ax.plot([sb_x, sb_x + 1.0], [sb_y, sb_y], color="#333", linewidth=3)
    ax.text(sb_x + 0.5, sb_y + 0.05 * (y_max - y_min) * 0.3,
            "1 m", ha="center", va="bottom", fontsize=9, color="#333")

    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Z (m)", fontsize=10)
    ax.set_title("Floor Plan — top-down projection", fontsize=13, fontweight="bold")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved floor plan → {out_path}")


def _mesh_planar_patch(
    pts_3d: np.ndarray,
    normal: np.ndarray,
    plane_d: float,
    max_edge_m: float = 0.30,
) -> tuple:
    """
    Project pts_3d onto their plane (normal · p + plane_d = 0),
    Delaunay-triangulate in 2-D, and remove any triangle with an edge
    longer than max_edge_m (alpha-shape filter — opens holes at
    windows / doors and unscanned regions).
    Returns (vertices_3d, triangles_int32) or (None, None).
    """
    from scipy.spatial import Delaunay

    if len(pts_3d) < 3:
        return None, None

    # Project onto the plane
    signed_d = pts_3d @ normal + plane_d
    pts_on   = pts_3d - signed_d[:, None] * normal

    # 2-D orthonormal basis on the plane
    ref = np.array([1., 0., 0.]) if abs(normal[0]) < 0.9 else np.array([0., 0., 1.])
    u   = np.cross(normal, ref);  u /= np.linalg.norm(u)
    v   = np.cross(normal, u);    v /= np.linalg.norm(v)
    p2d = np.column_stack([pts_on @ u, pts_on @ v])

    # Remove coincident projected points (voxel artefact)
    _, uid = np.unique(np.round(p2d, 4), axis=0, return_index=True)
    p2d    = p2d[uid]
    pts_on = pts_on[uid]
    if len(p2d) < 3:
        return None, None

    try:
        tri = Delaunay(p2d)
    except Exception:
        return None, None

    # Drop long triangles that span unscanned gaps (alpha-shape style)
    keep = [
        s for s in tri.simplices
        if (np.linalg.norm(pts_on[s[0]] - pts_on[s[1]]) <= max_edge_m and
            np.linalg.norm(pts_on[s[1]] - pts_on[s[2]]) <= max_edge_m and
            np.linalg.norm(pts_on[s[0]] - pts_on[s[2]]) <= max_edge_m)
    ]
    if not keep:
        return None, None
    return pts_on, np.array(keep, dtype=np.int32)

# ──────────────────────────────────────────────────────────────────────────────
# Stage4: Task 2 — Get elements like windows, doors, walls, floor, ceiling
# ──────────────────────────────────────────────────────────────────────────────
def find_architectural_elements(pcd: o3d.geometry.PointCloud, cfg: dict) -> dict:
    """
    Segment the dowsampled point cloud into architectural elements via
    iterative RANSAC plane detection followed by normal-based classification.

    Classification:
      |n · Y_world| > vertical_threshold   →  horizontal plane
          lowest centroid Y  →  floor
          highest centroid Y →  ceiling
      |n · Y_world| < horizontal_threshold →  wall
          Wall normals are snapped to the dominant 90°-spaced directions
          (rectilinear room assumption) then per-wall gap analysis runs:
            interior hole touching floor row  → door
            interior hole floating            → window

    Returns a dict  label → np.ndarray  of classified 3-D points.
    Saves a coloured PLY to  cfg["element_detection"]["output_path"].
    """
    from scipy.ndimage import binary_closing, binary_fill_holes, label as nd_label

    e_cfg = cfg.get("element_detection", {})
    out_path          = os.path.join(cfg["output_dir"],
                            e_cfg.get("output_path", "architectural_elements.ply"))
    max_planes        = e_cfg.get("max_planes",                20)
    min_inliers       = e_cfg.get("min_plane_inliers",         300)
    dist_thresh       = e_cfg.get("plane_distance_threshold",  0.04)
    vert_thresh       = e_cfg.get("normal_vertical_threshold",  0.85)
    horiz_thresh      = e_cfg.get("normal_horizontal_threshold",0.30)
    wall_res          = e_cfg.get("wall_resolution",            0.05)
    min_opening_area  = e_cfg.get("min_opening_area",           0.15)  # m²
    min_door_width    = e_cfg.get("min_door_width",             0.60)  # m
    max_door_width    = e_cfg.get("max_door_width",             1.80)  # m
    min_door_height   = e_cfg.get("min_door_height",            1.70)  # m

    Y_WORLD = np.array([0.0, 1.0, 0.0])

    COLORS = {
        "floor":        np.array([0.80, 0.65, 0.45]),   # warm sand
        "ceiling":      np.array([0.92, 0.92, 0.96]),   # cool white
        "wall":         np.array([0.75, 0.75, 0.78]),   # neutral grey
        "window":       np.array([0.35, 0.68, 0.95]),   # sky blue
        "door":         np.array([0.65, 0.38, 0.18]),   # wood brown
        "unclassified": np.array([0.30, 0.30, 0.30]),   # dark grey
    }

    all_points  = np.asarray(pcd.points).copy()   # (N, 3)
    all_colors  = np.asarray(pcd.colors).copy() if pcd.has_colors() else None
    n_total     = len(all_points)
    labels      = np.full(n_total, "unclassified", dtype=object)

    # ── 1. Iterative RANSAC plane detection ──────────────────────────────────
    remaining_idx = np.arange(n_total, dtype=np.int64)
    planes        = []   # list of (unit_normal, inlier_global_idx)

    work_pts = all_points.copy()
    work_pcd = o3d.geometry.PointCloud()
    work_pcd.points = o3d.utility.Vector3dVector(work_pts)

    for i in range(max_planes):
        if len(remaining_idx) < min_inliers:
            break
        model, local_inliers = work_pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000
        )
        if len(local_inliers) < min_inliers:
            break

        a, b, c, d_coeff = model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)

        global_inliers = remaining_idx[np.asarray(local_inliers)]
        planes.append((normal, global_inliers, model))

        keep = np.ones(len(remaining_idx), dtype=bool)
        keep[np.asarray(local_inliers)] = False
        remaining_idx = remaining_idx[keep]

        work_pts = all_points[remaining_idx]
        work_pcd = o3d.geometry.PointCloud()
        work_pcd.points = o3d.utility.Vector3dVector(work_pts)

    print(f"  detected {len(planes)} planes  ({len(remaining_idx):,} unclassified pts)")

    # ── 2. Classify planes by normal direction ────────────────────────────────
    horiz_planes = []   # (normal, inliers, centroid_y)
    wall_planes  = []   # (normal, inliers)

    for normal, inliers, raw_model in planes:
        y_dot = abs(float(np.dot(normal, Y_WORLD)))
        cy    = float(all_points[inliers, 1].mean())
        if y_dot > vert_thresh:
            horiz_planes.append((normal, inliers, cy, raw_model))
        elif y_dot < horiz_thresh:
            wall_planes.append((normal, inliers, raw_model))
        # else: slanted surface → stays unclassified

    # Floor = lowest horizontal plane; ceiling = highest
    floor_y = 0.0
    if horiz_planes:
        horiz_planes.sort(key=lambda x: x[2])
        labels[horiz_planes[0][1]]  = "floor"
        floor_y = float(horiz_planes[0][2])
        if len(horiz_planes) > 1:
            labels[horiz_planes[-1][1]] = "ceiling"
        for _, inliers, _, _ in horiz_planes[1:-1]:   # intermediate → ceiling
            labels[inliers] = "ceiling"
        print(f"  floor @y={horiz_planes[0][2]:.2f} m  "
              f"ceiling @y={horiz_planes[-1][2]:.2f} m  "
              f"({len(horiz_planes)} horizontal planes)")
    else:
        print("  [warn] no horizontal plane detected — floor_y assumed 0")

    for normal, inliers, _ in wall_planes:
        labels[inliers] = "wall"
    print(f"  {len(wall_planes)} wall planes")

    # ── 3. Snap wall normals to dominant 90°-spaced directions ───────────────
    if wall_planes:
        xz_normals = []
        for normal, _, _ in wall_planes:
            xz = np.array([normal[0], normal[2]])
            xz_norm = np.linalg.norm(xz)
            if xz_norm > 1e-6:
                xz_normals.append(xz / xz_norm)

        if xz_normals:
            angles = np.array([np.arctan2(n[1], n[0]) % np.pi
                               for n in xz_normals])
            hist, bins = np.histogram(angles, bins=180, range=(0.0, np.pi))
            peak_bin   = int(np.argmax(hist))
            theta0     = float((bins[peak_bin] + bins[peak_bin + 1]) / 2)
            print(f"  dominant wall directions: "
                  f"{np.degrees(theta0):.1f}° and "
                  f"{np.degrees(theta0 + np.pi/2):.1f}°  (XZ plane)")

    # ── 4. Per-wall gap analysis → windows and doors ──────────────────────────
    window_count = door_count = 0
    openings = []  # (model, u_wall, floor_y, hx_lo, hx_hi, hy_lo, hy_hi, lbl, wall_centroid)

    for wall_idx, (normal, inliers, wall_model) in enumerate(wall_planes):
        wall_pts = all_points[inliers]   # (K, 3)

        # Build wall-local 2D frame
        #   u = horizontal axis along wall face  (perpendicular to normal in XZ)
        #   v = world up (Y)
        n_xz = np.array([normal[0], 0.0, normal[2]])
        n_xz_len = float(np.linalg.norm(n_xz))
        if n_xz_len < 1e-6:
            continue
        n_xz /= n_xz_len
        u_wall = np.cross(Y_WORLD, n_xz)
        u_len  = float(np.linalg.norm(u_wall))
        if u_len < 1e-6:
            continue
        u_wall /= u_len

        x2d      = wall_pts @ u_wall          # horizontal position on wall face
        y_world  = wall_pts[:, 1]             # world height
        y_floor  = y_world - floor_y          # height above floor

        # Only use points above floor with some tolerance
        above = y_floor > -0.1
        if above.sum() < 50:
            continue
        x2d     = x2d[above]
        y_floor = y_floor[above]
        sub_idx = inliers[above]              # global indices of these points

        x_lo, x_hi = float(x2d.min()), float(x2d.max())
        y_lo_f = float(max(0.0, y_floor.min()))
        y_hi_f = float(y_floor.max())

        wall_w = x_hi - x_lo
        wall_h = y_hi_f - y_lo_f
        if wall_w < 0.4 or wall_h < 0.4:
            continue

        nx = max(int(wall_w / wall_res) + 1, 2)
        ny = max(int(wall_h / wall_res) + 1, 2)

        xi = np.clip(((x2d - x_lo) / wall_res).astype(int), 0, nx - 1)
        yi = np.clip(((y_floor - y_lo_f) / wall_res).astype(int), 0, ny - 1)

        grid = np.zeros((ny, nx), dtype=bool)
        grid[yi, xi] = True

        # Close scan noise (fills small gaps from LiDAR sparsity)
        grid   = binary_closing(grid, iterations=3)
        # Fill interior holes → solid wall
        filled = binary_fill_holes(grid)
        # Interior holes only
        holes  = filled & ~grid

        if not holes.any():
            continue

        labeled_holes, n_holes = nd_label(holes)

        for hole_id in range(1, n_holes + 1):
            hmask = labeled_holes == hole_id

            # Size filter
            hole_area = float(hmask.sum()) * wall_res * wall_res
            if hole_area < min_opening_area:
                continue

            rows_hit = np.where(hmask.any(axis=1))[0]
            cols_hit = np.where(hmask.any(axis=0))[0]
            hole_w = float(cols_hit.ptp() + 1) * wall_res
            hole_h = float(rows_hit.ptp() + 1) * wall_res

            # Reject holes that span most of the wall (likely unscanned region)
            if hole_w > wall_w * 0.85 or hole_h > wall_h * 0.85:
                continue

            # Bounding box in wall-local coordinates
            hx_lo = x_lo + float(cols_hit.min()) * wall_res
            hx_hi = x_lo + float(cols_hit.max() + 1) * wall_res
            hy_lo = y_lo_f + float(rows_hit.min()) * wall_res   # above floor
            hy_hi = y_lo_f + float(rows_hit.max() + 1) * wall_res

            # Re-label wall inliers inside this bounding box
            in_hole = (
                (x2d    >= hx_lo) & (x2d    <= hx_hi) &
                (y_floor >= hy_lo) & (y_floor <= hy_hi)
            )
            global_in_hole = sub_idx[in_hole]

            # Classify: hole touching row-0 (floor level) with door dimensions → door;
            # otherwise → window.
            at_floor     = bool(hmask[0, :].any())
            looks_like_door = (
                at_floor
                and min_door_width <= hole_w <= max_door_width
                and hole_h >= min_door_height
            )

            if looks_like_door:
                labels[global_in_hole] = "door"
                door_count += 1
                print(f"  wall {wall_idx}: door  {door_count}  "
                      f"{hole_w:.2f} m wide × {hole_h:.2f} m tall")
                openings.append((wall_model, u_wall.copy(), floor_y,
                                 hx_lo, hx_hi, hy_lo, hy_hi, "door",
                                 wall_pts.mean(0)))
            else:
                labels[global_in_hole] = "window"
                window_count += 1
                print(f"  wall {wall_idx}: window {window_count}  "
                      f"{hole_w:.2f} m wide × {hole_h:.2f} m tall")
                openings.append((wall_model, u_wall.copy(), floor_y,
                                 hx_lo, hx_hi, hy_lo, hy_hi, "window",
                                 wall_pts.mean(0)))

    print(f"  → {window_count} window(s),  {door_count} door(s) detected")

    # ── 5. Assemble coloured output cloud & save ─────────────────────────────
    out_colors = np.zeros((n_total, 3))
    for lbl, col in COLORS.items():
        out_colors[labels == lbl] = col
    if all_colors is not None:
        unc = labels == "unclassified"
        out_colors[unc] = all_colors[unc] * 0.45

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(all_points)
    out_pcd.colors = o3d.utility.Vector3dVector(out_colors)
    o3d.io.write_point_cloud(out_path, out_pcd)
    print(f"  saved point cloud  → {out_path}")
    _save_gif(out_path, cfg)

    # ── 6. Build & save triangulated mesh ──────────────────────────────────
    mesh_path   = os.path.join(cfg["output_dir"],
                      e_cfg.get("mesh_output_path", "architectural_elements_mesh.ply"))
    max_edge    = float(e_cfg.get("mesh_max_edge",   0.30))   # alpha-shape edge limit (m)
    opening_off = float(e_cfg.get("opening_offset",  0.01))   # m nudge off wall plane

    all_verts_list: list = []
    all_tris_list:  list = []
    all_vcols_list: list = []
    v_off = 0

    def _add_to_mesh(verts, tris, color):
        nonlocal v_off
        if verts is None or tris is None or len(verts) == 0 or len(tris) == 0:
            return
        n_v   = len(verts)
        vcols = np.tile(color, (n_v, 1))
        dtris = np.vstack([tris, tris[:, ::-1]])   # double-sided winding
        all_verts_list.append(verts.astype(np.float64))
        all_tris_list.append((dtris + v_off).astype(np.int32))
        all_vcols_list.append(vcols)
        v_off += n_v

    Y_WLD = np.array([0., 1., 0.])

    # --- Floor / ceiling patches ---
    for i_hp, (pnorm, pinliers, pcy, pmodel) in enumerate(horiz_planes):
        lbl  = "floor" if i_hp == 0 else "ceiling"
        mask = labels[pinliers] == lbl
        pts  = all_points[pinliers[mask]]
        if len(pts) < 3:
            continue
        pa, pb, pc, pd = pmodel
        pnrm = np.array([pa, pb, pc]);  pnrm /= np.linalg.norm(pnrm)
        verts, tris = _mesh_planar_patch(pts, pnrm, float(pd), max_edge)
        _add_to_mesh(verts, tris, COLORS[lbl])
        if verts is not None:
            print(f"  meshed {lbl:8s}: {len(verts):,} verts  {len(tris):,} tris")

    # --- Wall patches (window/door points excluded → physical holes in wall mesh) ---
    for (pnorm, pinliers, pmodel) in wall_planes:
        mask = labels[pinliers] == "wall"
        pts  = all_points[pinliers[mask]]
        if len(pts) < 3:
            continue
        pa, pb, pc, pd = pmodel
        pnrm = np.array([pa, pb, pc]);  pnrm /= np.linalg.norm(pnrm)
        verts, tris = _mesh_planar_patch(pts, pnrm, float(pd), max_edge)
        _add_to_mesh(verts, tris, COLORS["wall"])
        if verts is not None:
            print(f"  meshed wall    : {len(verts):,} verts  {len(tris):,} tris")

    # --- Window / door solid quads (fill the holes, offset ± for double-face) ---
    for (o_model, o_uwl, o_fy, ox_lo, ox_hi, oy_lo, oy_hi, o_lbl, o_wcen) in openings:
        oa, ob, oc, od = o_model
        o_nrm = np.array([oa, ob, oc]);  o_nrm /= np.linalg.norm(o_nrm)
        # Reference point on the wall plane
        P0     = o_wcen - (float(o_wcen @ o_nrm) + float(od)) * o_nrm
        x0_ref = float(P0 @ o_uwl)
        y0_ref = float(P0[1]) - o_fy
        q_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        for sign in (+1.0, -1.0):
            nudge   = sign * opening_off * o_nrm
            corners = np.array([
                P0 + (ox_lo - x0_ref) * o_uwl + (oy_lo - y0_ref) * Y_WLD + nudge,
                P0 + (ox_hi - x0_ref) * o_uwl + (oy_lo - y0_ref) * Y_WLD + nudge,
                P0 + (ox_hi - x0_ref) * o_uwl + (oy_hi - y0_ref) * Y_WLD + nudge,
                P0 + (ox_lo - x0_ref) * o_uwl + (oy_hi - y0_ref) * Y_WLD + nudge,
            ])
            _add_to_mesh(corners, q_tris, COLORS[o_lbl])
        print(f"  mesh quad {o_lbl:6s}: {ox_hi - ox_lo:.2f} m × {oy_hi - oy_lo:.2f} m")

    # --- Assemble & write ---
    if all_verts_list:
        m_verts = np.vstack(all_verts_list)
        m_tris  = np.vstack(all_tris_list)
        m_cols  = np.vstack(all_vcols_list)
        out_mesh = o3d.geometry.TriangleMesh()
        out_mesh.vertices      = o3d.utility.Vector3dVector(m_verts)
        out_mesh.triangles     = o3d.utility.Vector3iVector(m_tris)
        out_mesh.vertex_colors = o3d.utility.Vector3dVector(m_cols)
        out_mesh.remove_degenerate_triangles()
        o3d.io.write_triangle_mesh(mesh_path, out_mesh)
        print(f"  saved mesh         → {mesh_path}  "
              f"({len(out_mesh.vertices):,} verts, "
              f"{len(out_mesh.triangles):,} tris)")
        _save_gif(mesh_path, cfg)
    else:
        print("  [warn] no mesh produced — check min_plane_inliers or scan coverage")

    print("")
    for lbl in ("floor", "ceiling", "wall", "window", "door", "unclassified"):
        count = int((labels == lbl).sum())
        pct   = 100.0 * count / n_total
        print(f"  {lbl:15s}: {count:7,} pts  ({pct:5.1f}%)")

    return {lbl: all_points[labels == lbl] for lbl in COLORS}


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    log_path = _init_logger(output_dir)
    # Print to the tee immediately so the path appears in the log too
    print(f"Log: {log_path}")
    print(f"Run started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {cfg['dataset_path']}")
    print(f"Output dir: {output_dir}")

    proc_h = cfg["processing_height"]
    proc_w = cfg["processing_width"]

    pipeline_start = time.time()

    # Stage 1 — back-project & fuse
    t0 = time.time()
    pcd = stage1_backproject(cfg, proc_h, proc_w)
    print(f"  [Stage 1 time: {_elapsed(time.time() - t0)}]")

    # Stage 2 — downsample / filter
    t0 = time.time()
    downsampled_pcd = stage2_downsample(pcd, cfg)
    print(f"  [Stage 2 time: {_elapsed(time.time() - t0)}]")

    # Stage 3 — surface reconstruction
    t0 = time.time()
    mesh = stage3_reconstruct(downsampled_pcd, cfg)
    print(f"  [Stage 3 time: {_elapsed(time.time() - t0)}]")

    # Stage 4.1 — floor plan (optional)
    if cfg.get("floor_plan", {}).get("enabled", False):
        print("\n[Stage 4.1] Extracting floor plan …")
        t0 = time.time()
        find_floor_plan(downsampled_pcd, cfg)
        print(f"  [Stage 4.1 time: {_elapsed(time.time() - t0)}]")

    # Stage 4.2 — architectural element detection (optional)
    if cfg.get("element_detection", {}).get("enabled", False):
        print("\n[Stage 4.2] Detecting architectural elements …")
        t0 = time.time()
        find_architectural_elements(downsampled_pcd, cfg)
        print(f"  [Stage 4.2 time: {_elapsed(time.time() - t0)}]")

    print(f"\nTotal pipeline time: {_elapsed(time.time() - pipeline_start)}")
    print(f"Run finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nDone.")

    # Optional: open3d visualiser (after log is fully written)
    vis_cfg = cfg.get("visualize", {})
    if vis_cfg.get("enabled", False):
        sys.stdout.flush()
        o3d.visualization.draw_geometries(
            [downsampled_pcd],
            window_name="Point Cloud",
            width=vis_cfg.get("window_width", 1280),
            height=vis_cfg.get("window_height", 720),
        )


if __name__ == "__main__":
    main()
