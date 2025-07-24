#!/usr/bin/env python
"""
map_and_show_path.py
Run with:  python map_and_show_path.py scan_back.ply
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import argparse
from pathlib import Path
from typing import Tuple

from scipy.signal import savgol_filter  

DEFAULT_PLY = Path("pointclouds/clean_back_scan.ply")

# ---- copy your zig‑zag generator here (or import it) ----
def generate_zigzag(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    num_points: int = 100,
    base_offset_factor: float = 0.2,
    amplitude_factor: float = 0.15,
    frequency: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates left and right zigzag paths within a bounding box using a triangular waveform.

    Parameters:
        min_x, max_x, min_y, max_y : float
            Boundaries of the 2D space.
        num_points : int
            Total number of points (shared between left and right paths).
        base_offset_factor : float
            Controls the horizontal offset of the base path from the center.
        amplitude_factor : float
            Controls the horizontal amplitude of the zigzag pattern.
        frequency : int
            Number of zigzag cycles along the vertical axis.

    Returns:
        points_left, points_right : Tuple[np.ndarray, np.ndarray]
            Arrays of shape (N, 2) containing (x, y) points for the left and right zigzag paths.
    """

    width = max_x - min_x
    height = max_y - min_y
    center_x = (min_x + max_x) / 2

    base_offset = width * base_offset_factor
    amplitude = width * amplitude_factor

    base_x_left = center_x - base_offset
    base_x_right = center_x + base_offset

    # Determine minimum required points to preserve zigzag shape
    points_needed_min = (int(frequency * 2) + 1) * 2
    points_per_path = max(points_needed_min, num_points // 2)

    # Construct vertical zigzag path: up then down
    y_up = np.linspace(min_y, max_y, points_per_path // 2)
    y_down = np.linspace(max_y, min_y, points_per_path // 2)
    y_path = np.concatenate((y_up, y_down))

    # Normalize y to range [0, 1] for waveform generation
    norm_y = (y_path - min_y) / height
    phase = (frequency * norm_y) % 1.0

    # Generate triangle waveform for zigzag
    tri_wave = np.where(phase < 0.5,
                        (phase * 4) - 1,         # rising edge
                        ((1.0 - phase) * 4) - 1) # falling edge

    # Apply horizontal zigzag oscillation
    x_oscillation = amplitude * tri_wave
    x_left = np.clip(base_x_left - x_oscillation, min_x, max_x)
    x_right = np.clip(base_x_right + x_oscillation, min_x, max_x)

    # Combine into final point arrays
    points_left = np.column_stack((x_left, y_path))
    points_right = np.column_stack((x_right, y_path))

    return points_left, points_right


# ---- 2‑D → 3‑D mapping helper ----
def map_2d_path_to_3d(pcd, path_xy, radius=3.0):
    xyz   = np.asarray(pcd.points)
    tree  = cKDTree(xyz[:, :2])                # 2‑D KD‑tree (X,Y)
    pts3d = []
    for xy in path_xy:
        idxs = tree.query_ball_point(xy, r=radius)
        if not idxs:                           # hole → nearest neighbour
            _, idx = tree.query(xy)
            idxs = [idx]
        z_val = xyz[idxs, 2].mean()
        pts3d.append([xy[0], xy[1], z_val])
    return np.asarray(pts3d)


# ---- build a coloured LineSet from a path ----
def as_lineset(path_3d, rgb):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(path_3d)
    ls.lines  = o3d.utility.Vector2iVector(
        np.column_stack([np.arange(len(path_3d)-1),
                         np.arange(1, len(path_3d))]))
    ls.colors = o3d.utility.Vector3dVector([rgb]*(len(path_3d)-1))
    return ls


def offset_path_along_normals(pcd: o3d.geometry.PointCloud,
                              path: np.ndarray,
                              offset: float) -> np.ndarray:
    """
    Offsets each point in *path* by *offset* along the nearest cloud normal.
    """
    if offset == 0:
        return path

    # Ensure we have normals
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=50)
        )

    xyz = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)

    # Nearest-neighbour on full 3D (path is already on/near the surface)
    tree3 = cKDTree(xyz)
    _, idxs = tree3.query(path, k=1)

    path_off = path + nrm[idxs] * offset
    return path_off

def detect_spine(
    pcd: o3d.geometry.PointCloud,
    min_x: float, max_x: float,
    min_y: float, max_y: float,
    bins_y: int = 200,
    corridor_frac: float = 0.25,   # corridor width as fraction of box width
    z_mode: str = "min",           # "min" = valley (lower than around), "max" = ridge
    resample_radius: float = 5.0,  # XY radius to (re)sample Z after straightening
    smooth: bool = True,
    smooth_win_frac: float = 0.1,  # % of samples, auto-odd
    min_pts_per_slice: int = 20
) -> np.ndarray:
    """
    Find an almost-straight spine line inside the given XY box, assuming the spine
    is a shallow Z *valley* (z_mode='min') or *ridge* (z_mode='max').

    Returns: (K, 3) ndarray ordered by Y.
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError("Empty point cloud")

    # restrict to the XY box
    m = (pts[:,0] >= min_x) & (pts[:,0] <= max_x) & \
        (pts[:,1] >= min_y) & (pts[:,1] <= max_y)
    box = pts[m]
    if box.shape[0] == 0:
        raise RuntimeError("No points inside XY box")

    # centre corridor
    cx = 0.5 * (min_x + max_x)
    corridor_w = (max_x - min_x) * corridor_frac

    y_edges = np.linspace(min_y, max_y, bins_y + 1)
    picks = []
    choose = np.argmin if z_mode == "min" else np.argmax

    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        slab = box[(box[:,1] >= y0) & (box[:,1] < y1) &
                   (np.abs(box[:,0] - cx) <= corridor_w * 0.5)]
        if slab.shape[0] < min_pts_per_slice:
            continue
        idx = choose(slab[:,2])
        picks.append(slab[idx])

    if len(picks) == 0:
        raise RuntimeError("No candidates found. Try widening corridor_frac, lowering min_pts_per_slice, or switching z_mode.")

    picks = np.asarray(picks)
    # order by Y
    order = np.argsort(picks[:,1])
    picks = picks[order]

    # ---- fit straight line x = a*y + b on the picks (least squares) ----
    y = picks[:,1]
    x = picks[:,0]
    A = np.vstack([y, np.ones_like(y)]).T
    a, b = np.linalg.lstsq(A, x, rcond=None)[0]
    x_line = a * y + b

    # ---- resample Z along that (almost straight) line from the original cloud ----
    tree_xy = cKDTree(pts[:, :2])
    path3d = []
    for yi, xi in zip(y, x_line):
        ids = tree_xy.query_ball_point([xi, yi], r=resample_radius)
        if not ids:
            # fallback to nearest
            _, idx = tree_xy.query([xi, yi])
            ids = [idx]
        zs = pts[ids, 2]
        z_val = np.min(zs) if z_mode == "min" else np.max(zs)
        path3d.append([xi, yi, z_val])

    path3d = np.asarray(path3d)

    # ---- optional smoothing ----
    if smooth and len(path3d) > 7:
        win = max(5, int(len(path3d) * smooth_win_frac) | 1)
        for c in range(3):
            path3d[:, c] = savgol_filter(path3d[:, c], window_length=win, polyorder=3)

    return path3d

# --------------- MAIN ---------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", type=Path, nargs="?", default=DEFAULT_PLY, help="back scan .ply")
    ap.add_argument("--radius", type=float, default=5.0,
                    help="XY search radius (same units as scan)")
    ap.add_argument(
    "--no-cloud", action="store_true",
    help="Hide the point‑cloud and show only the massage path",
    )
    ap.add_argument("--path-offset", action="store_true",
                help="Push the 3D path outwards along surface normals by this distance (same units as the scan).")
    args = ap.parse_args()

    # 1. load scan
    pcd = o3d.io.read_point_cloud(str(args.scan))
    if pcd.is_empty():
        raise RuntimeError("Empty point cloud!")

    # 2. create 2‑D zig‑zag path (edit bounds to match your back)
    #    ↳ here we auto‑derive X & Y bounds from the scan
    xyz = np.asarray(pcd.points)
    # min_x, max_x = np.percentile(xyz[:,0], [2, 98])   # crop out outliers
    # min_y, max_y = np.percentile(xyz[:,1], [ 5, 95])
    min_x, max_x = 450, 1300
    min_y, max_y = -200, 600

    pts_left2d, pts_right2d = generate_zigzag(
                        min_x, max_x, min_y, max_y,
                        num_points=100, frequency=6
                        )
    # 3. map to 3‑D
    path_left3d  = map_2d_path_to_3d(pcd, pts_left2d,  radius=args.radius)
    path_right3d = map_2d_path_to_3d(pcd, pts_right2d, radius=args.radius)
    
    if args.path_offset:
        offset = 10 # same units  as the pointcloud, path should be visible with this setup
        path_left3d  = offset_path_along_normals(pcd, path_left3d, offset)
        path_right3d = offset_path_along_normals(pcd, path_right3d, offset)

    # 4. visualise
    geoms = []
    spine_pts = detect_spine(
        pcd,
        min_x, max_x, min_y, max_y,
        bins_y=300,
        corridor_frac=0.20,  
        z_mode="min",         
        resample_radius=7.0,
        smooth=True
    )

    spine_pts = offset_path_along_normals(pcd, spine_pts, offset=10.0)
    spine_ls  = as_lineset(spine_pts, rgb=(0, 0, 0))   # red spine
    geoms.append(spine_ls) 

    if not args.no_cloud:
        geoms.append(pcd)                 # add cloud only if wanted
    geoms.append(as_lineset(path_left3d, [0, 0, 0]))     # path always shown
    geoms.append(as_lineset(path_right3d, [0, 0, 0]))     # path always shown
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Back scan + massage path",
                    width=1280, height=720)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.line_width = 150.0              #  ❰❰  thickness in pixels
    opt.background_color = np.array([1.0, 1.0, 1.0])  # white bg helps contrast

    vis.run()
    vis.destroy_window()