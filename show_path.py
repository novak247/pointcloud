#!/usr/bin/env python
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import argparse
from pathlib import Path
from typing import Tuple
import time

from scipy.signal import savgol_filter  
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

DEFAULT_PLY = Path("pointclouds/clean_back_scan.ply")

def extrapolate_spine_region(spine_left: np.ndarray, 
                           spine_right: np.ndarray,
                           full_y_min: float, 
                           full_y_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrapolate spine boundaries to cover the full back range
    
    Parameters:
    - spine_left/right: detected spine boundary points (M, 3)
    - full_y_min/max: full Y range that needs to be covered
    
    Returns:
    - extended_spine_left/right: spine boundaries covering full range
    """
    
    if len(spine_left) < 3 or len(spine_right) < 3:
        print("Warning: Too few spine points for reliable extrapolation")
        return spine_left, spine_right
    
    # Sort spine points by Y coordinate
    left_sorted = spine_left[np.argsort(spine_left[:, 1])]
    right_sorted = spine_right[np.argsort(spine_right[:, 1])]
    
    # Get current Y range of detected spine
    detected_y_min = min(left_sorted[0, 1], right_sorted[0, 1])
    detected_y_max = max(left_sorted[-1, 1], right_sorted[-1, 1])
    
    print(f"Detected spine Y range: {detected_y_min:.1f} to {detected_y_max:.1f}")
    print(f"Full back Y range: {full_y_min:.1f} to {full_y_max:.1f}")
    
    extended_left = left_sorted.copy()
    extended_right = right_sorted.copy()
    
    # === EXTRAPOLATE BACKWARDS (towards full_y_min) ===
    if detected_y_min > full_y_min:
        print(f"Extrapolating spine backwards by {detected_y_min - full_y_min:.1f} units")
        
        # Use first 3 points to determine slope
        left_slope_points = left_sorted[:3]
        right_slope_points = right_sorted[:3]
        
        # Calculate average slopes (delta_x / delta_y) for extrapolation
        left_dx_dy = np.mean(np.diff(left_slope_points[:, 0]) / np.diff(left_slope_points[:, 1]))
        left_dz_dy = np.mean(np.diff(left_slope_points[:, 2]) / np.diff(left_slope_points[:, 1]))
        
        right_dx_dy = np.mean(np.diff(right_slope_points[:, 0]) / np.diff(right_slope_points[:, 1]))
        right_dz_dy = np.mean(np.diff(right_slope_points[:, 2]) / np.diff(right_slope_points[:, 1]))
        
        # Create extrapolated points
        num_backward_points = max(3, int((detected_y_min - full_y_min) / 10))  # One point per ~10 units
        y_backward = np.linspace(full_y_min, detected_y_min - 1, num_backward_points)
        
        left_backward = []
        right_backward = []
        
        for y_new in y_backward:
            dy = y_new - left_sorted[0, 1]  # Distance from first detected point
            
            left_x_new = left_sorted[0, 0] + left_dx_dy * dy
            left_z_new = left_sorted[0, 2] + left_dz_dy * dy
            left_backward.append([left_x_new, y_new, left_z_new])
            
            right_x_new = right_sorted[0, 0] + right_dx_dy * dy
            right_z_new = right_sorted[0, 2] + right_dz_dy * dy
            right_backward.append([right_x_new, y_new, right_z_new])
        
        # Prepend to existing points
        extended_left = np.vstack([np.array(left_backward), extended_left])
        extended_right = np.vstack([np.array(right_backward), extended_right])
    
    # === EXTRAPOLATE FORWARDS (towards full_y_max) ===
    if detected_y_max < full_y_max:
        print(f"Extrapolating spine forwards by {full_y_max - detected_y_max:.1f} units")
        
        # Use last 3 points to determine slope
        left_slope_points = left_sorted[-3:]
        right_slope_points = right_sorted[-3:]
        
        # Calculate average slopes
        left_dx_dy = np.mean(np.diff(left_slope_points[:, 0]) / np.diff(left_slope_points[:, 1]))
        left_dz_dy = np.mean(np.diff(left_slope_points[:, 2]) / np.diff(left_slope_points[:, 1]))
        
        right_dx_dy = np.mean(np.diff(right_slope_points[:, 0]) / np.diff(right_slope_points[:, 1]))
        right_dz_dy = np.mean(np.diff(right_slope_points[:, 2]) / np.diff(right_slope_points[:, 1]))
        
        # Create extrapolated points
        num_forward_points = max(3, int((full_y_max - detected_y_max) / 10))
        y_forward = np.linspace(detected_y_max + 1, full_y_max, num_forward_points)
        
        left_forward = []
        right_forward = []
        
        for y_new in y_forward:
            dy = y_new - left_sorted[-1, 1]  # Distance from last detected point
            
            left_x_new = left_sorted[-1, 0] + left_dx_dy * dy
            left_z_new = left_sorted[-1, 2] + left_dz_dy * dy
            left_forward.append([left_x_new, y_new, left_z_new])
            
            right_x_new = right_sorted[-1, 0] + right_dx_dy * dy
            right_z_new = right_sorted[-1, 2] + right_dz_dy * dy
            right_forward.append([right_x_new, y_new, right_z_new])
        
        # Append to existing points
        extended_left = np.vstack([extended_left, np.array(left_forward)])
        extended_right = np.vstack([extended_right, np.array(right_forward)])
    
    print(f"Extended spine: {len(extended_left)} points covering Y {extended_left[0,1]:.1f} to {extended_left[-1,1]:.1f}")
    
    return extended_left, extended_right

def create_extended_spine_mesh(extended_left: np.ndarray, 
                              extended_right: np.ndarray,
                              detected_left: np.ndarray,
                              detected_right: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    Create spine mesh with different colors for detected vs extrapolated regions
    """
    
    n_points = len(extended_left)
    if n_points < 2:
        return None
    
    # Create vertices
    vertices = np.vstack([extended_left, extended_right])
    
    # Create faces
    faces = []
    for i in range(n_points - 1):
        faces.append([i, n_points + i, i + 1])
        faces.append([n_points + i, n_points + i + 1, i + 1])
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Color coding: detected region vs extrapolated
    vertex_colors = np.zeros((len(vertices), 3))
    
    # Find which points are detected vs extrapolated
    detected_y_min = np.min(detected_left[:, 1])
    detected_y_max = np.max(detected_left[:, 1])
    
    for i, vertex in enumerate(vertices):
        y_coord = vertex[1]
        if detected_y_min <= y_coord <= detected_y_max:
            vertex_colors[i] = [1.0, 0.3, 0.3]  # Red for detected spine
        else:
            vertex_colors[i] = [1.0, 0.7, 0.7]  # Light red for extrapolated
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.compute_vertex_normals()
    
    return mesh

def check_spine_collision_extended(zigzag_points: np.ndarray, 
                                 extended_spine_left: np.ndarray, 
                                 extended_spine_right: np.ndarray,
                                 safety_margin: float = 20.0) -> np.ndarray:
    """
    Check collision with extended spine region that covers full back
    """
    
    if len(extended_spine_left) < 2 or len(extended_spine_right) < 2:
        return np.zeros(len(zigzag_points), dtype=bool)
    
    # Create interpolation functions for the extended spine
    spine_y_coords = extended_spine_left[:, 1]
    spine_left_x = extended_spine_left[:, 0]
    spine_right_x = extended_spine_right[:, 0]
    
    # Sort by Y coordinate
    sort_indices = np.argsort(spine_y_coords)
    spine_y_sorted = spine_y_coords[sort_indices]
    spine_left_x_sorted = spine_left_x[sort_indices]
    spine_right_x_sorted = spine_right_x[sort_indices]
    
    try:
        left_interp = interp1d(spine_y_sorted, spine_left_x_sorted, 
                              kind='linear', bounds_error=False, fill_value='extrapolate')
        right_interp = interp1d(spine_y_sorted, spine_right_x_sorted, 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
    except Exception as e:
        print(f"Interpolation failed: {e}")
        return np.zeros(len(zigzag_points), dtype=bool)
    
    # Check each zigzag point
    collision_mask = np.zeros(len(zigzag_points), dtype=bool)
    collision_count = 0
    
    for i, (x, y) in enumerate(zigzag_points):
        # Get spine boundaries at this Y coordinate (including extrapolated regions)
        try:
            left_boundary = left_interp(y) - safety_margin
            right_boundary = right_interp(y) + safety_margin
            
            # Check if point is between boundaries (inside spine region)
            if left_boundary <= x <= right_boundary:
                collision_mask[i] = True
                collision_count += 1
        except:
            continue  # Skip if interpolation fails for this point
    
    print(f"Collision check: {collision_count}/{len(zigzag_points)} points collide with extended spine")
    return collision_mask

def adjust_zigzag_for_extended_spine(zigzag_points: np.ndarray,
                                   extended_spine_left: np.ndarray,
                                   extended_spine_right: np.ndarray,
                                   safety_margin: float = 20.0,
                                   is_left_path: bool = True) -> np.ndarray:
    """
    Adjust zigzag points to avoid collision with extended spine
    """
    
    if len(extended_spine_left) < 2 or len(extended_spine_right) < 2:
        return zigzag_points
    
    # Create interpolation functions
    spine_y_coords = extended_spine_left[:, 1]
    spine_left_x = extended_spine_left[:, 0]
    spine_right_x = extended_spine_right[:, 0]
    
    sort_indices = np.argsort(spine_y_coords)
    spine_y_sorted = spine_y_coords[sort_indices]
    spine_left_x_sorted = spine_left_x[sort_indices]
    spine_right_x_sorted = spine_right_x[sort_indices]
    
    try:
        left_interp = interp1d(spine_y_sorted, spine_left_x_sorted, 
                              kind='linear', bounds_error=False, fill_value='extrapolate')
        right_interp = interp1d(spine_y_sorted, spine_right_x_sorted, 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
    except:
        return zigzag_points
    
    adjusted_points = zigzag_points.copy()
    collision_count = 0
    
    for i, (x, y) in enumerate(zigzag_points):
        try:
            left_boundary = left_interp(y)
            right_boundary = right_interp(y)
            
            # Check for collision and adjust
            if is_left_path:
                # Left path should stay left of spine
                required_x = left_boundary - safety_margin
                if x > required_x:  # Collision - move further left
                    adjusted_points[i, 0] = required_x
                    collision_count += 1
            else:
                # Right path should stay right of spine  
                required_x = right_boundary + safety_margin
                if x < required_x:  # Collision - move further right
                    adjusted_points[i, 0] = required_x
                    collision_count += 1
        except:
            continue  # Skip if interpolation fails
    
    if collision_count > 0:
        print(f"Adjusted {collision_count} points in {'left' if is_left_path else 'right'} zigzag for extended spine avoidance")
    
    return adjusted_points
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

def check_spine_collision(zigzag_points: np.ndarray, 
                         spine_left: np.ndarray, 
                         spine_right: np.ndarray,
                         safety_margin: float = 20.0) -> np.ndarray:
    """
    Check which zigzag points collide with the spine region
    
    Parameters:
    - zigzag_points: (N, 2) array of (x, y) zigzag points
    - spine_left: (M, 3) array of left spine boundary points
    - spine_right: (M, 3) array of right spine boundary points  
    - safety_margin: additional clearance around spine
    
    Returns:
    - collision_mask: boolean array indicating which points collide
    """
    
    if len(spine_left) < 2 or len(spine_right) < 2:
        return np.zeros(len(zigzag_points), dtype=bool)
    
    # Create interpolation functions for spine boundaries
    spine_y_coords = spine_left[:, 1]
    spine_left_x = spine_left[:, 0]
    spine_right_x = spine_right[:, 0]
    
    # Sort by Y coordinate for interpolation
    sort_indices = np.argsort(spine_y_coords)
    spine_y_sorted = spine_y_coords[sort_indices]
    spine_left_x_sorted = spine_left_x[sort_indices]
    spine_right_x_sorted = spine_right_x[sort_indices]
    
    # Create interpolation functions
    try:
        left_interp = interp1d(spine_y_sorted, spine_left_x_sorted, 
                              kind='linear', bounds_error=False, fill_value='extrapolate')
        right_interp = interp1d(spine_y_sorted, spine_right_x_sorted, 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
    except:
        # Fallback if interpolation fails
        return np.zeros(len(zigzag_points), dtype=bool)
    
    # Check each zigzag point
    collision_mask = np.zeros(len(zigzag_points), dtype=bool)
    
    for i, (x, y) in enumerate(zigzag_points):
        # Get spine boundaries at this Y coordinate
        left_boundary = left_interp(y) - safety_margin
        right_boundary = right_interp(y) + safety_margin
        
        # Check if point is between boundaries (inside spine region)
        if left_boundary <= x <= right_boundary:
            collision_mask[i] = True
    
    return collision_mask

def adjust_zigzag_for_spine_avoidance(zigzag_points: np.ndarray,
                                    spine_left: np.ndarray,
                                    spine_right: np.ndarray,
                                    safety_margin: float = 20.0,
                                    is_left_path: bool = True) -> np.ndarray:
    """
    Adjust zigzag points to avoid spine collision
    
    Parameters:
    - zigzag_points: (N, 2) array of zigzag points
    - spine_left/right: spine boundary points
    - safety_margin: clearance from spine
    - is_left_path: True for left zigzag, False for right zigzag
    
    Returns:
    - adjusted_points: (N, 2) array of adjusted zigzag points
    """
    
    if len(spine_left) < 2 or len(spine_right) < 2:
        return zigzag_points
    
    # Create interpolation functions
    spine_y_coords = spine_left[:, 1]
    spine_left_x = spine_left[:, 0]
    spine_right_x = spine_right[:, 0]
    
    sort_indices = np.argsort(spine_y_coords)
    spine_y_sorted = spine_y_coords[sort_indices]
    spine_left_x_sorted = spine_left_x[sort_indices]
    spine_right_x_sorted = spine_right_x[sort_indices]
    
    try:
        left_interp = interp1d(spine_y_sorted, spine_left_x_sorted, 
                              kind='linear', bounds_error=False, fill_value='extrapolate')
        right_interp = interp1d(spine_y_sorted, spine_right_x_sorted, 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
    except:
        return zigzag_points
    
    adjusted_points = zigzag_points.copy()
    collision_count = 0
    
    for i, (x, y) in enumerate(zigzag_points):
        left_boundary = left_interp(y)
        right_boundary = right_interp(y)
        
        # Check for collision and adjust
        if is_left_path:
            # Left path should stay left of spine
            required_x = left_boundary - safety_margin
            if x > required_x:  # Collision - move further left
                adjusted_points[i, 0] = required_x
                collision_count += 1
        else:
            # Right path should stay right of spine  
            required_x = right_boundary + safety_margin
            if x < required_x:  # Collision - move further right
                adjusted_points[i, 0] = required_x
                collision_count += 1
    
    if collision_count > 0:
        print(f"Adjusted {collision_count} points in {'left' if is_left_path else 'right'} zigzag to avoid spine")
    
    return adjusted_points

def smooth_adjusted_zigzag(adjusted_points: np.ndarray, 
                          original_points: np.ndarray,
                          smoothing_factor: float = 0.3) -> np.ndarray:
    """
    Smooth the adjusted zigzag to maintain natural curves
    
    Parameters:
    - adjusted_points: points after spine avoidance
    - original_points: original zigzag points
    - smoothing_factor: how much to smooth (0=no smooth, 1=full smooth)
    
    Returns:
    - smoothed_points: smoothed zigzag points
    """
    
    if len(adjusted_points) < 5:
        return adjusted_points
    
    smoothed = adjusted_points.copy()
    
    # Apply smoothing filter to X coordinates only (preserve Y)
    try:
        window_length = min(7, len(adjusted_points) - 1)
        if window_length >= 5 and window_length % 2 == 1:  # Must be odd
            x_smooth = savgol_filter(adjusted_points[:, 0], 
                                   window_length=window_length, 
                                   polyorder=3)
            
            # Blend with original for natural curves
            smoothed[:, 0] = (smoothing_factor * x_smooth + 
                            (1 - smoothing_factor) * adjusted_points[:, 0])
    except:
        pass  # Keep original if smoothing fails
    
    return smoothed


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

# ========== IMPROVED FAST SPINE DETECTION ==========

def smart_downsample_for_spine(points, target_size=50000):
    """
    Smart downsampling that preserves spine structure
    """
    if len(points) <= target_size:
        return points
    
    # Use stratified sampling to preserve distribution
    # Sample more densely in the center corridor where spine is likely
    center_x = np.median(points[:, 0])
    distances_from_center = np.abs(points[:, 0] - center_x)
    
    # Higher probability for points closer to center
    probabilities = 1.0 / (1.0 + distances_from_center / np.std(distances_from_center))
    probabilities = probabilities / probabilities.sum()
    
    # Sample points
    indices = np.random.choice(len(points), size=target_size, replace=False, p=probabilities)
    return points[indices]

def create_spine_rectangle(spine_centerline, spine_width=50.0):
    """
    Create a rectangular spine region from the centerline
    
    Parameters:
    - spine_centerline: (N, 3) array of spine center points
    - spine_width: width of the spine region in same units as point cloud
    
    Returns:
    - spine_left: left boundary of spine region
    - spine_right: right boundary of spine region  
    - spine_mesh: mesh representing the spine area
    """
    
    if len(spine_centerline) < 2:
        return None, None, None
    
    # Calculate spine direction vectors
    spine_directions = np.diff(spine_centerline, axis=0)
    # Add the last direction to match array size
    spine_directions = np.vstack([spine_directions, spine_directions[-1]])
    
    # Normalize directions
    spine_lengths = np.linalg.norm(spine_directions, axis=1)
    spine_lengths[spine_lengths == 0] = 1  # Avoid division by zero
    spine_unit_dirs = spine_directions / spine_lengths.reshape(-1, 1)
    
    # Calculate perpendicular vectors (in XY plane)
    # Perpendicular to spine direction, pointing left/right
    perp_vectors = np.zeros_like(spine_unit_dirs)
    perp_vectors[:, 0] = -spine_unit_dirs[:, 1]  # -dy
    perp_vectors[:, 1] = spine_unit_dirs[:, 0]   # dx
    perp_vectors[:, 2] = 0  # Keep in XY plane
    
    # Create left and right boundaries
    half_width = spine_width * 0.5
    spine_left = spine_centerline + perp_vectors * half_width
    spine_right = spine_centerline - perp_vectors * half_width
    
    return spine_left, spine_right, perp_vectors

def detect_spine_boundaries_in_slices(
    pcd: o3d.geometry.PointCloud,
    spine_centerline: np.ndarray,
    search_radius: float = 30.0,
    z_mode: str = "min"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each point in spine_centerline, find the actual left/right boundaries
    of the spine region by analyzing cross-sections
    """
    
    pts = np.asarray(pcd.points)
    tree = cKDTree(pts)
    
    spine_left = []
    spine_right = []
    
    print(f"Detecting spine boundaries for {len(spine_centerline)} points...")
    
    for i, center_point in enumerate(spine_centerline):
        # Get points near this spine location
        nearby_indices = tree.query_ball_point(center_point, r=search_radius)
        
        if len(nearby_indices) < 10:
            # Fallback: use fixed width
            spine_left.append([center_point[0] - 25, center_point[1], center_point[2]])
            spine_right.append([center_point[0] + 25, center_point[1], center_point[2]])
            continue
            
        nearby_points = pts[nearby_indices]
        
        # Focus on points at similar Y coordinate (cross-section)
        y_tolerance = 15.0  # Adjust based on your data resolution
        cross_section_mask = np.abs(nearby_points[:, 1] - center_point[1]) < y_tolerance
        cross_section = nearby_points[cross_section_mask]
        
        if len(cross_section) < 5:
            # Fallback
            spine_left.append([center_point[0] - 25, center_point[1], center_point[2]])
            spine_right.append([center_point[0] + 25, center_point[1], center_point[2]])
            continue
        
        # Find spine region boundaries based on Z values
        if z_mode == "min":
            # For valley: find the extent of the low region
            z_threshold = np.percentile(cross_section[:, 2], 30)  # Bottom 30%
            spine_candidates = cross_section[cross_section[:, 2] <= z_threshold]
        else:
            # For ridge: find the extent of the high region  
            z_threshold = np.percentile(cross_section[:, 2], 70)  # Top 30%
            spine_candidates = cross_section[cross_section[:, 2] >= z_threshold]
        
        if len(spine_candidates) >= 3:
            # Find leftmost and rightmost points of spine region
            x_coords = spine_candidates[:, 0]
            left_x = np.min(x_coords)
            right_x = np.max(x_coords)
            
            # Get Z values at boundaries
            left_z = np.mean(spine_candidates[x_coords == left_x, 2])
            right_z = np.mean(spine_candidates[x_coords == right_x, 2])
            
            spine_left.append([left_x, center_point[1], left_z])
            spine_right.append([right_x, center_point[1], right_z])
        else:
            # Fallback to fixed width
            spine_left.append([center_point[0] - 25, center_point[1], center_point[2]])
            spine_right.append([center_point[0] + 25, center_point[1], center_point[2]])
    
    return np.array(spine_left), np.array(spine_right)

def create_spine_mesh(spine_left, spine_right):
    """
    Create a mesh surface representing the spine region
    """
    n_points = len(spine_left)
    if n_points < 2:
        return None
    
    # Create vertices: left boundary + right boundary
    vertices = np.vstack([spine_left, spine_right])
    
    # Create triangular faces connecting left and right boundaries
    faces = []
    for i in range(n_points - 1):
        # Two triangles per quad segment
        # Triangle 1: (left_i, right_i, left_i+1)
        faces.append([i, n_points + i, i + 1])
        # Triangle 2: (right_i, right_i+1, left_i+1)  
        faces.append([n_points + i, n_points + i + 1, i + 1])
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Color the mesh
    mesh.paint_uniform_color([1.0, 0.5, 0.5])  # Light red
    mesh.compute_vertex_normals()
    
    return mesh

def detect_spine_region(
    pcd: o3d.geometry.PointCloud,
    min_x: float, max_x: float,
    min_y: float, max_y: float,
    bins_y: int = 200,
    corridor_frac: float = 0.25,
    z_mode: str = "min",
    resample_radius: float = 5.0,
    smooth: bool = True,
    smooth_win_frac: float = 0.1,
    min_pts_per_slice: int = 20,
    spine_detection_method: str = "adaptive"  # "adaptive" or "fixed_width"
) -> dict:
    """
    Detect spine region returning centerline, boundaries, and mesh
    
    Returns dict with:
    - centerline: spine center points
    - left_boundary: left edge of spine
    - right_boundary: right edge of spine  
    - mesh: 3D mesh of spine region
    """
    
    print("Starting spine region detection...")
    start_time = time.time()
    
    # First, get the centerline using existing method
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError("Empty point cloud")
    
    # [Your existing centerline detection code here - abbreviated for space]
    # This is the same logic as before to find the spine centerline
    box_mask = ((pts[:,0] >= min_x) & (pts[:,0] <= max_x) & 
                (pts[:,1] >= min_y) & (pts[:,1] <= max_y))
    box_points = pts[box_mask]
    
    if len(box_points) > 100000:
        # Smart downsample
        center_x = np.median(box_points[:, 0])
        distances_from_center = np.abs(box_points[:, 0] - center_x)
        probabilities = 1.0 / (1.0 + distances_from_center / np.std(distances_from_center))
        probabilities = probabilities / probabilities.sum()
        indices = np.random.choice(len(box_points), size=80000, replace=False, p=probabilities)
        box_points = box_points[indices]
    
    cx = 0.5 * (min_x + max_x)
    corridor_w = (max_x - min_x) * corridor_frac
    y_edges = np.linspace(min_y, max_y, bins_y + 1)
    picks = []
    choose = np.argmin if z_mode == "min" else np.argmax

    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        y_mask = (box_points[:,1] >= y0) & (box_points[:,1] < y1)
        y_slice = box_points[y_mask]
        if len(y_slice) == 0:
            continue
        corridor_mask = np.abs(y_slice[:,0] - cx) <= corridor_w * 0.5
        slab = y_slice[corridor_mask]
        if slab.shape[0] < min_pts_per_slice:
            continue
        idx = choose(slab[:,2])
        picks.append(slab[idx])

    if len(picks) == 0:
        raise RuntimeError("No spine centerline found")

    picks = np.asarray(picks)
    order = np.argsort(picks[:,1])
    picks = picks[order]
    
    # Remove outliers
    if len(picks) > 10:
        x_coords = picks[:,0]
        x_median = np.median(x_coords)
        x_mad = np.median(np.abs(x_coords - x_median))
        outlier_threshold = max(3.0 * x_mad, (max_x - min_x) * 0.05)
        good_mask = np.abs(x_coords - x_median) <= outlier_threshold
        picks = picks[good_mask]

    # Fit line and resample Z
    y = picks[:,1]
    x = picks[:,0]
    A = np.vstack([y, np.ones_like(y)]).T
    try:
        a, b = np.linalg.lstsq(A, x, rcond=None)[0]
        x_line = a * y + b
    except:
        x_line = x

    tree_xy = cKDTree(pts[:, :2])
    centerline = []
    for yi, xi in zip(y, x_line):
        ids = tree_xy.query_ball_point([xi, yi], r=resample_radius)
        if not ids:
            _, idx = tree_xy.query([xi, yi])
            ids = [idx]
        zs = pts[ids, 2]
        z_val = np.min(zs) if z_mode == "min" else np.max(zs)
        centerline.append([xi, yi, z_val])

    centerline = np.asarray(centerline)

    # Smooth centerline
    if smooth and len(centerline) > 7:
        win = max(5, int(len(centerline) * smooth_win_frac) | 1)
        win = min(win, len(centerline) - 1)
        if win >= 5 and win < len(centerline):
            try:
                for c in range(3):
                    centerline[:, c] = savgol_filter(centerline[:, c], window_length=win, polyorder=3)
            except:
                pass

    # Now detect spine boundaries
    if spine_detection_method == "adaptive":
        # Detect actual boundaries from the point cloud
        spine_left, spine_right = detect_spine_boundaries_in_slices(
            pcd, centerline, search_radius=40.0, z_mode=z_mode
        )
    else:
        # Use fixed width
        spine_left, spine_right, _ = create_spine_rectangle(centerline, spine_width=50.0)
    
    # Create mesh
    spine_mesh = create_spine_mesh(spine_left, spine_right)
    
    total_time = time.time() - start_time
    print(f"Spine region detection completed in {total_time:.3f}s")
    print(f"Centerline: {len(centerline)} points")
    print(f"Spine region width: ~{np.mean(np.linalg.norm(spine_right - spine_left, axis=1)):.1f} units")
    
    return {
        'centerline': centerline,
        'left_boundary': spine_left,
        'right_boundary': spine_right,
        'mesh': spine_mesh
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", type=Path, nargs="?", default=DEFAULT_PLY, help="back scan .ply")
    ap.add_argument("--radius", type=float, default=5.0,
                    help="XY search radius (same units as scan)")
    ap.add_argument("--no-cloud", action="store_true",
                help="Hide the point‑cloud and show only the massage path")
    ap.add_argument("--path-offset", action="store_true",
                help="Push the 3D path outwards along surface normals")
    ap.add_argument("--spine-margin", type=float, default=25.0,
                help="Safety margin around spine (same units as scan)")
    ap.add_argument("--show-original", action="store_true",
                help="Show original zigzag paths for comparison")
    ap.add_argument("--show-extended", action="store_true",
                help="Show extended spine region for debugging")
    args = ap.parse_args()

    # Load scan
    pcd = o3d.io.read_point_cloud(str(args.scan))
    if pcd.is_empty():
        raise RuntimeError("Empty point cloud!")

    # Define bounds
    min_x, max_x = 450, 1300
    min_y, max_y = -200, 600

    # 1. DETECT SPINE REGION FIRST
    print("=== DETECTING SPINE REGION ===")
    try:
        spine_result = detect_spine_region(
            pcd, min_x, max_x, min_y, max_y,
            bins_y=300,
            corridor_frac=0.20,
            z_mode="min",
            resample_radius=7.0,
            smooth=True,
            spine_detection_method="adaptive"
        )
        
        detected_spine_left = spine_result['left_boundary']
        detected_spine_right = spine_result['right_boundary']
        
        # 2. EXTRAPOLATE SPINE TO COVER FULL BACK
        print("=== EXTRAPOLATING SPINE REGION ===")
        extended_spine_left, extended_spine_right = extrapolate_spine_region(
            detected_spine_left, detected_spine_right, min_y, max_y
        )
        
    except Exception as e:
        print(f"Spine detection failed: {e}")
        detected_spine_left = detected_spine_right = None
        extended_spine_left = extended_spine_right = None

    # 3. GENERATE INITIAL ZIGZAG PATHS
    print("=== GENERATING ZIGZAG PATHS ===")
    pts_left2d_original, pts_right2d_original = generate_zigzag(
                        min_x, max_x, min_y, max_y,
                        num_points=100, frequency=6)

    # 4. ADJUST ZIGZAG PATHS TO AVOID EXTENDED SPINE
    if extended_spine_left is not None and extended_spine_right is not None:
        print("=== ADJUSTING PATHS FOR EXTENDED SPINE AVOIDANCE ===")
        
        # Check for collisions with extended spine
        left_collisions = check_spine_collision_extended(
            pts_left2d_original, extended_spine_left, extended_spine_right, args.spine_margin)
        right_collisions = check_spine_collision_extended(
            pts_right2d_original, extended_spine_left, extended_spine_right, args.spine_margin)
        
        print(f"Extended spine collision check:")
        print(f"  Left path collisions: {np.sum(left_collisions)}/{len(left_collisions)}")
        print(f"  Right path collisions: {np.sum(right_collisions)}/{len(right_collisions)}")
        
        # Adjust paths using extended spine
        pts_left2d_adjusted = adjust_zigzag_for_extended_spine(
            pts_left2d_original, extended_spine_left, extended_spine_right, 
            safety_margin=args.spine_margin, is_left_path=True)
        
        pts_right2d_adjusted = adjust_zigzag_for_extended_spine(
            pts_right2d_original, extended_spine_left, extended_spine_right,
            safety_margin=args.spine_margin, is_left_path=False)
        
        # Smooth the adjusted paths
        pts_left2d_final = smooth_adjusted_zigzag(pts_left2d_adjusted, pts_left2d_original)
        pts_right2d_final = smooth_adjusted_zigzag(pts_right2d_adjusted, pts_right2d_original)
        
    else:
        print("Using original zigzag paths (no spine detected)")
        pts_left2d_final = pts_left2d_original
        pts_right2d_final = pts_right2d_original

    # 5. MAP TO 3D
    print("=== MAPPING TO 3D ===")
    path_left3d = map_2d_path_to_3d(pcd, pts_left2d_final, radius=args.radius)
    path_right3d = map_2d_path_to_3d(pcd, pts_right2d_final, radius=args.radius)
    
    if args.path_offset:
        offset = 10
        path_left3d = offset_path_along_normals(pcd, path_left3d, offset)
        path_right3d = offset_path_along_normals(pcd, path_right3d, offset)

    # 6. VISUALIZE
    print("=== VISUALIZING ===")
    geoms = []
    
    # Add extended spine region if detected
    if extended_spine_left is not None and extended_spine_right is not None:
        if args.show_extended:
            # Show extended spine mesh with color coding
            extended_spine_mesh = create_extended_spine_mesh(
                extended_spine_left, extended_spine_right,
                detected_spine_left, detected_spine_right
            )
            
            if extended_spine_mesh is not None:
                vertices = np.asarray(extended_spine_mesh.vertices)
                vertices_offset = offset_path_along_normals(pcd, vertices, offset=10.0)
                extended_spine_mesh.vertices = o3d.utility.Vector3dVector(vertices_offset)
                geoms.append(extended_spine_mesh)
            
            # Add boundary lines for extended spine
            extended_left_offset = offset_path_along_normals(pcd, extended_spine_left, offset=10.0)
            extended_right_offset = offset_path_along_normals(pcd, extended_spine_right, offset=10.0)
            
            geoms.append(as_lineset(extended_left_offset, [1, 0, 0]))     # Red boundaries
            geoms.append(as_lineset(extended_right_offset, [1, 0, 0]))
        
        else:
            # Show only the detected spine region (original)
            if detected_spine_left is not None:
                detected_left_offset = offset_path_along_normals(pcd, detected_spine_left, offset=10.0)
                detected_right_offset = offset_path_along_normals(pcd, detected_spine_right, offset=10.0)
                
                geoms.append(as_lineset(detected_left_offset, [1, 0, 0]))
                geoms.append(as_lineset(detected_right_offset, [1, 0, 0]))

    # Add original paths for comparison (if requested)
    if args.show_original:
        path_left3d_orig = map_2d_path_to_3d(pcd, pts_left2d_original, radius=args.radius)
        path_right3d_orig = map_2d_path_to_3d(pcd, pts_right2d_original, radius=args.radius)
        if args.path_offset:
            path_left3d_orig = offset_path_along_normals(pcd, path_left3d_orig, offset)
            path_right3d_orig = offset_path_along_normals(pcd, path_right3d_orig, offset)
        
        geoms.append(as_lineset(path_left3d_orig, [0.7, 0.7, 0.7]))   # Gray original
        geoms.append(as_lineset(path_right3d_orig, [0.7, 0.7, 0.7]))

    # Add adjusted paths
    geoms.append(as_lineset(path_left3d, [0, 0, 0]))     # Blue adjusted left
    geoms.append(as_lineset(path_right3d, [0, 0, 0]))    # Green adjusted right

    if not args.no_cloud:
        geoms.append(pcd)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Extended Spine-Avoiding Massage Paths", width=1280, height=720)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.line_width = 150.0
    opt.background_color = np.array([1.0, 1.0, 1.0])

    vis.run()
    vis.destroy_window()
