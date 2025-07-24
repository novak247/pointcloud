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

DEFAULT_PLY = Path.home() / "skola"/ "pointcloud"/ "pointclouds" / "pairwise_icp_pair_0_5.ply"

# ---- copy your zig‑zag generator here (or import it) ----
def generate_zigzag(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    num_points: int = 50,
    base_offset_factor: float = 0.2,
    amplitude_factor: float = 0.15,
    frequency: int = 5
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
    args = ap.parse_args()

    # 1. load scan
    pcd = o3d.io.read_point_cloud(str(args.scan))
    if pcd.is_empty():
        raise RuntimeError("Empty point cloud!")

    # 2. create 2‑D zig‑zag path (edit bounds to match your back)
    #    ↳ here we auto‑derive X & Y bounds from the scan
    xyz = np.asarray(pcd.points)
    min_x, max_x = np.percentile(xyz[:,0], [2, 98])   # crop out outliers
    min_y, max_y = np.percentile(xyz[:,1], [ 5, 95])

    pts_left2d, pts_right2d = generate_zigzag(
                        min_x, max_x, min_y, max_y,
                        num_points=80, frequency=6
                        )
    # 3. map to 3‑D
    path_left3d  = map_2d_path_to_3d(pcd, pts_left2d,  radius=args.radius)
    path_right3d = map_2d_path_to_3d(pcd, pts_right2d, radius=args.radius)

    # 4. visualise
    geoms = []
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
    opt.line_width = 50.0              #  ❰❰  thickness in pixels
    opt.background_color = np.array([1.0, 1.0, 1.0])  # white bg helps contrast

    vis.run()
    vis.destroy_window()