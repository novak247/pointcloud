#!/usr/bin/env python
"""
Quick .ply point cloud viewer using Open3D.

Usage:
    python view_ply.py path/to/cloud.ply [--voxel 0.01] [--normals]
"""
import argparse
import numpy as np
import open3d as o3d
import sys
from pathlib import Path
import math
import json


DEFAULT_PLY = Path("pointclouds/clean_back_scan.ply")

def load_point_cloud(path: Path):
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"No points loaded from {path}")
    return pcd

def color_by_height(pcd):
    # normalize z to 0-1 and map to jet colormap
    pts = np.asarray(pcd.points)
    z = pts[:, 2]
    z_min, z_max = z.min(), z.max()
    if z_max > z_min:
        z_norm = (z - z_min) / (z_max - z_min)
    else:
        z_norm = np.zeros_like(z)
    cmap = plt_jet(z_norm)  # Nx3 float in [0,1]
    pcd.colors = o3d.utility.Vector3dVector(cmap)

def plt_jet(vals):
    """Simple jet colormap without importing matplotlib; vals in [0,1]."""
    # crude manual mapping; for nicer colors you can import matplotlib
    import numpy as np
    v = np.clip(vals, 0, 1)
    r = np.clip(1.5 - np.abs(4*v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4*v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4*v - 1), 0, 1)
    return np.stack([r, g, b], axis=1)

def estimate_normals(pcd, radius=0.05, max_nn=30):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.normalize_normals()
    
def make_axis_ticks(min_b, max_b, spacing, font_size, color):
    """
    Return a list of Open3D geometries (LineSets + Text3D) that form
    ticks & numeric labels around the axis‑aligned box defined by min_b/max_b.
    """
    geoms = []
    axes = ['X', 'Y', 'Z']
    min_b = np.asarray(min_b)
    max_b = np.asarray(max_b)
    for ax in range(3):                       # 0,1,2 → X,Y,Z
        rng = max_b[ax] - min_b[ax]
        if rng == 0:            # degenerate
            continue
        # default spacing ≈ 10 ticks
        step = spacing or (rng / 10.0)
        # round to neat value (1, 2, 5 × 10^n)
        if spacing is None:
            mag = 10 ** np.floor(np.log10(step))
            step = np.round(step / mag)
            step = 1 if step < 1.5 else 2 if step < 3.5 else 5
            step *= mag

        ticks = np.arange(np.floor(min_b[ax] / step) * step,
                          np.ceil(max_b[ax] / step) * step + 1e-6, step)

        for t in ticks:
            # line: small “tick” perpendicular to axis
            p0 = min_b.copy()
            p1 = min_b.copy()
            p0[ax] = p1[ax] = t
            ortho = (ax + 1) % 3               # pick another axis for 1‑cm tick
            tick_len = 0.01 * rng if rng else 0.01
            p1[ortho] += tick_len

            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([p0, p1]),
                lines=o3d.utility.Vector2iVector([[0, 1]]),
            )
            ls.paint_uniform_color(color)
            geoms.append(ls)

            geoms.append(make_text(f"{t:.2f}".rstrip('0').rstrip('.'), p1, font_size, color, axis=ax))
    return geoms


def make_text(message, pos, height, color, axis=0, two_sided=True):
    """
    Build a label mesh facing the viewer, rotated so its baseline aligns
    with the chosen *axis* (0=X, 1=Y, 2=Z).
    """
    # -- create unit text mesh (faces -Z) --
    mesh = o3d.t.geometry.TriangleMesh.create_text(message, depth=0.001) \
                                       .to_legacy()

    # -- scale to requested height --
    bbox    = mesh.get_axis_aligned_bounding_box()
    h_unit  = bbox.get_extent()[1] or 1.0
    mesh.scale(height / h_unit, center=bbox.get_center())

    # -- orient baseline along desired world axis --
    rot = {
        0: (math.pi/2,0,math.pi/2),   # X axis → rotate -90° about Z
        1: (0,0,0),           # Y axis → rotate  90° about X
        2: (math.pi/2, 0, 0),       # Z axis → default (faces +Z)
    }[axis]
    mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz(rot),
                center=mesh.get_center())


    # -- move into place & colour --
    mesh.translate(pos - mesh.get_center())
    mesh.paint_uniform_color(color)
    return mesh


def load_transform(json_path: Path, invert: bool = False) -> np.ndarray:
    with open(json_path, "r") as f:
        data = json.load(f)

    if "T" not in data:
        raise KeyError(f"'T' not found in {json_path}")

    T = np.asarray(data["T"], dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"Transform in {json_path} must be 4x4, got {T.shape}")

    if invert:
        T = np.linalg.inv(T)

    return T


def main():
    parser = argparse.ArgumentParser(description="Visualize a .ply point cloud.")
    parser.add_argument("ply_path", type=Path, nargs="?", default=DEFAULT_PLY, help=".ply file to load")
    parser.add_argument("--voxel", type=float, default=None,
                        help="Voxel size for optional downsampling (e.g. 0.01 meters).")
    parser.add_argument("--normals", action="store_true",
                        help="Estimate and display normals (slower).")
    parser.add_argument("--height-colors", action="store_true",
                        help="Color points by height (z-axis).")
    parser.add_argument("--save-screenshot", type=Path, default=None,
                        help="Path to save a screenshot (png) after window closes.")
    parser.add_argument("--axes",action="store_true",help="Show an XYZ coordinate frame (default size 0.2 m)")
    parser.add_argument("--axes-size",type=float,default=0.2,
                        help="Size of the coordinate frame in the same units as the point cloud (default: 0.2)",)
    parser.add_argument("--axes-rel", action="store_true",
                        help="Place the axes at the point-cloud centre instead of the world origin")
    parser.add_argument("--axes-scale", type=float, default=None,
                        help="Manual scale for axes; if omitted I'll auto-scale to 10 % of the cloud")
    parser.add_argument("--ticks", type=float, default=None,
                        help="Tick spacing for axis labels. "
                         "If omitted, I’ll auto‑pick ≈ 10 ticks per axis.")
    parser.add_argument("--tick-font", type=int, default=20,
                        help="Font size for tick labels (Open3D Text3D).")
    parser.add_argument("--tick-color", type=float, nargs=3, default=[0.8, 0.8, 0.8],
                        metavar=("R", "G", "B"),
                        help="RGB in [0,1] for tick lines and text (default light‑grey).")
    parser.add_argument("--transform", type=Path, default=None,
                        help="Path to a JSON file containing a 4x4 matrix under the key 'T'.")
    parser.add_argument("--invert-transform", action="store_true",
                        help="Invert the transform before applying (use if your T is world→sensor instead of sensor→world).")
    args = parser.parse_args()

    if not args.ply_path.is_file():
        sys.exit(f"File not found: {args.ply_path}")

    print("Loading point cloud...")
    pcd = load_point_cloud(args.ply_path)
    print(pcd)
    if args.transform is not None:
        T = load_transform(args.transform, invert=args.invert_transform)
        pcd.transform(T)   # rotates normals too, if present

    if args.voxel:
        print(f"Downsampling with voxel size {args.voxel}...")
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel)
        print("After downsampling:", pcd)

    if args.normals:
        print("Estimating normals...")
        estimate_normals(pcd)

    if args.height_colors:
        print("Coloring by height...")
        color_by_height(pcd)

    # Show oriented bounding box (optional)
    obb = pcd.get_oriented_bounding_box()
    obb.color = (1, 0, 0)

    geoms = [pcd, obb]

    if args.axes:
        # --- decide size ---
        if args.axes_scale is not None:
            size = args.axes_scale
        else:
            diag = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
            size = 0.1 * diag           # 10 % of the bounding‑box diagonal

        # --- decide origin ---
        if args.axes_rel:
            origin = pcd.get_center()   # put axes at the cloud centre
        else:
            origin = [0, 0, 0]

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size,
                                                                origin=origin)
        geoms.append(coord)
    
    if args.ticks is not None or args.ticks == None:   # user asked OR auto default
        aabb = pcd.get_axis_aligned_bounding_box()          # world‑aligned
        tick_geoms = make_axis_ticks(aabb.get_min_bound(),
                                    aabb.get_max_bound(),
                                    spacing=args.ticks,
                                    font_size=args.tick_font,
                                    color=args.tick_color)
        geoms.extend(tick_geoms)
        
    print("Launching viewer (close the window to end)...")
    o3d.visualization.draw_geometries(
        geoms,
        window_name=str(args.ply_path),
        point_show_normal=args.normals,
        mesh_show_back_face=True,
    )

    if args.save_screenshot:
        # Offscreen render & save
        print(f"Saving screenshot to {args.save_screenshot} ...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        vis.add_geometry(obb)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(args.save_screenshot), do_render=True)
        vis.destroy_window()
        print("Done.")

if __name__ == "__main__":
    main()
