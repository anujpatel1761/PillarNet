import numpy as np

def load_lidar_file(file_path):
    """
    Load lidar point cloud from binary file.
    
    Args:
        file_path (str): Path to the .bin file
    
    Returns:
        np.ndarray: Point cloud array of shape (N, 4) with [x, y, z, intensity]
    """
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points

def filter_point_cloud_range(points, x_range=(0, 70.4), y_range=(-40, 40), z_range=(-3, 1)):
    """
    Filter point cloud based on detection range as specified in PointPillars paper.
    
    Args:
        points (np.ndarray): Point cloud of shape (N, 4) with [x, y, z, intensity]
        x_range (tuple): Range for x coordinates (default: car range)
        y_range (tuple): Range for y coordinates (default: car range)  
        z_range (tuple): Range for z coordinates (default: car range)
    
    Returns:
        np.ndarray: Filtered point cloud
    """
    # Create boolean mask for each dimension
    x_mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
    y_mask = (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    z_mask = (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    
    # Combine all masks
    valid_mask = x_mask & y_mask & z_mask
    
    # Filter points
    filtered_points = points[valid_mask]
    
    print(f"Original points: {len(points)}, Filtered points: {len(filtered_points)}")
    return filtered_points