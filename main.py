"""
Example usage of the PointPillars preprocessing pipeline.

This demonstrates the complete "Point Cloud to Pseudo-Image" conversion process
as described in the PointPillars paper.
"""

from preprocess.point_cloud_loader import load_lidar_file, filter_point_cloud_range
from preprocess.pillar_creator import point_cloud_to_pillars  
from preprocess.dense_tensor_creator import create_dense_tensor, create_pseudo_image_indices

def main():
    # Configuration for car detection (from PointPillars paper)
    CAR_CONFIG = {
        'x_range': (0, 70.4),       # meters
        'y_range': (-40, 40),       # meters  
        'z_range': (-3, 1),         # meters
        'grid_size_x': 0.16,        # meters
        'grid_size_y': 0.16,        # meters
        'max_pillars': 12000,       # P parameter
        'max_points_per_pillar': 100 # N parameter
    }
    
    print("=== PointPillars Preprocessing Pipeline ===\n")
    
    # Step 1: Load and filter point cloud
    print("Step 1: Loading and filtering point cloud...")
    file_path = "path/to/your/pointcloud.bin"  # Replace with actual path
    
    # For demo, create synthetic data if file doesn't exist
    try:
        points = load_lidar_file(file_path)
    except:
        print("Creating synthetic point cloud for demo...")
        # Create synthetic point cloud for demonstration
        np.random.seed(42)
        points = np.random.rand(50000, 4).astype(np.float32)
        points[:, 0] *= 100  # x: 0-100
        points[:, 1] *= 80 - 40  # y: -40 to 40
        points[:, 2] *= 10 - 5   # z: -5 to 5
        points[:, 3] *= 255      # intensity: 0-255
    
    # Filter points to detection range
    filtered_points = filter_point_cloud_range(
        points, 
        x_range=CAR_CONFIG['x_range'],
        y_range=CAR_CONFIG['y_range'], 
        z_range=CAR_CONFIG['z_range']
    )
    
    # Step 2: Convert to pillars with 9D augmentation
    print(f"\nStep 2: Converting to pillars...")
    augmented_pillars, pillar_centers = point_cloud_to_pillars(
        filtered_points,
        grid_size_x=CAR_CONFIG['grid_size_x'],
        grid_size_y=CAR_CONFIG['grid_size_y']
    )
    
    # Step 3: Create dense tensor with sparsity handling
    print(f"\nStep 3: Creating dense tensor...")
    dense_tensor, pillar_coordinates, filled_pillars = create_dense_tensor(
        augmented_pillars,
        max_pillars=CAR_CONFIG['max_pillars'],
        max_points_per_pillar=CAR_CONFIG['max_points_per_pillar']
    )
    
    # Step 4: Prepare for scattering back to pseudo-image
    print(f"\nStep 4: Preparing pseudo-image indices...")
    
    # Calculate pseudo-image dimensions
    x_range = CAR_CONFIG['x_range'][1] - CAR_CONFIG['x_range'][0]
    y_range = CAR_CONFIG['y_range'][1] - CAR_CONFIG['y_range'][0]
    image_width = int(x_range / CAR_CONFIG['grid_size_x'])   # W
    image_height = int(y_range / CAR_CONFIG['grid_size_y'])  # H
    
    batch_indices, y_indices, x_indices = create_pseudo_image_indices(
        pillar_coordinates, filled_pillars, image_height, image_width
    )
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Dense tensor ready for PointNet: {dense_tensor.shape}")
    print(f"Pseudo-image dimensions: ({image_height}, {image_width})")
    print(f"Ready for 2D CNN backbone!")
    
    return {
        'dense_tensor': dense_tensor,
        'pillar_coordinates': pillar_coordinates,
        'scatter_indices': (batch_indices, y_indices, x_indices),
        'pseudo_image_size': (image_height, image_width),
        'filled_pillars': filled_pillars
    }

if __name__ == "__main__":
    import numpy as np
    results = main()