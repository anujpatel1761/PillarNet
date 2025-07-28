import numpy as np
import random

def point_cloud_to_pillars(points, grid_size_x=0.16, grid_size_y=0.16):
    """
    Convert point cloud to pillar representation with 9D augmented features.
    
    This function implements the core pillar creation logic from PointPillars paper:
    1. Discretize point cloud into evenly spaced grid in x-y plane
    2. Group points by pillar coordinates
    3. Calculate pillar centers and arithmetic means
    4. Augment each point with relative distances (xc, yc, zc, xp, yp)
    
    Args:
        points (np.ndarray): Point cloud of shape (N, 4) with [x, y, z, intensity]
        grid_size_x (float): Grid resolution in x direction (default: 0.16m)
        grid_size_y (float): Grid resolution in y direction (default: 0.16m)
    
    Returns:
        dict: Dictionary with pillar_id as key and list of 9D augmented points as value
        dict: Dictionary with pillar_id as key and pillar center coordinates as value
    """
    pillar_data = {}
    
    # Step 1: Group points into pillars using grid discretization
    for point in points:
        x, y, z, intensity = point
        
        # Calculate pillar indices using floor division
        pillar_x = int(np.floor(x / grid_size_x))
        pillar_y = int(np.floor(y / grid_size_y))
        pillar_id = (pillar_x, pillar_y)
        
        # Group points by pillar
        if pillar_id not in pillar_data:
            pillar_data[pillar_id] = []
        pillar_data[pillar_id].append([x, y, z, intensity])
    
    print(f"Created {len(pillar_data)} non-empty pillars from {len(points)} points")
    
    # Step 2: Calculate pillar centers and augment points with 9D features
    augmented_pillars = {}
    pillar_centers = {}
    
    for pillar_id, points_in_pillar in pillar_data.items():
        points_array = np.array(points_in_pillar)
        
        # Calculate pillar center (geometric center of grid cell)
        pillar_center_x = pillar_id[0] * grid_size_x + (grid_size_x / 2)
        pillar_center_y = pillar_id[1] * grid_size_y + (grid_size_y / 2)
        pillar_centers[pillar_id] = (pillar_center_x, pillar_center_y)
        
        # Calculate arithmetic mean of all points in this pillar
        mean_x = np.mean(points_array[:, 0])
        mean_y = np.mean(points_array[:, 1]) 
        mean_z = np.mean(points_array[:, 2])
        
        # Step 3: Create 9D augmented features for each point
        augmented_points = []
        for point in points_array:
            x, y, z, intensity = point
            
            # Distance to arithmetic mean (c = cluster/mean)
            xc = x - mean_x
            yc = y - mean_y
            zc = z - mean_z
            
            # Distance to pillar center (p = pillar)
            xp = x - pillar_center_x
            yp = y - pillar_center_y
            
            # Create 9D augmented point: [x, y, z, intensity, xc, yc, zc, xp, yp]
            augmented_point = [x, y, z, intensity, xc, yc, zc, xp, yp]
            augmented_points.append(augmented_point)
        
        augmented_pillars[pillar_id] = augmented_points
    
    return augmented_pillars, pillar_centers