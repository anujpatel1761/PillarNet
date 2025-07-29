"""
main_pipeline.py

Clean PointPillars preprocessing pipeline implementation.
Pipeline: Point Cloud → Filter → Pillars → Dense Tensor → PointNet → Pseudo-Image
"""

import torch
import numpy as np

from preprocess.point_cloud_loader import load_lidar_file, filter_point_cloud_range
from preprocess.pillar_creator import point_cloud_to_pillars  
from preprocess.dense_tensor_creator import create_dense_tensor
from preprocess.pointnet_encoder import PointNetEncoder, scatter_to_pseudo_image


def main():
    # PointPillars configuration for car detection
    CONFIG = {
        'x_range': (0, 70.4), 'y_range': (-40, 40), 'z_range': (-3, 1),
        'grid_size_x': 0.16, 'grid_size_y': 0.16,
        'max_pillars': 12000, 'max_points_per_pillar': 100, 'feature_channels': 64
    }
    
    print("=== PointPillars Pipeline ===")
    
    # Step 1: Load and filter point cloud
    file_path = "C:/Users/anujp/Desktop/PillarNet/data/000000.bin"
    points = load_lidar_file(file_path)
    filtered_points = filter_point_cloud_range(points, **{k: v for k, v in CONFIG.items() if 'range' in k})
    print(f"Points: {len(points)} → {len(filtered_points)} (filtered)")

    # Step 2: Convert to pillars
    pillars, _ = point_cloud_to_pillars(filtered_points, CONFIG['grid_size_x'], CONFIG['grid_size_y'])
    print(f"Pillars: {len(pillars)} non-empty")
    
    # Step 3: Create dense tensor
    dense_tensor, pillar_coords, filled_pillars = create_dense_tensor(
        pillars, CONFIG['max_pillars'], CONFIG['max_points_per_pillar']
    )
    print(f"Dense tensor: {dense_tensor.shape}")
    
    # Step 4: PointNet encoding
    encoder = PointNetEncoder(in_channels=9, out_channels=CONFIG['feature_channels'])
    dense_tensor = torch.from_numpy(dense_tensor).float()
    
    # Ensure correct shape: (D, P, N)
    if dense_tensor.shape[0] != 9:
        dense_tensor = dense_tensor.permute(2, 0, 1)
    
    with torch.no_grad():
        pillar_features = encoder(dense_tensor)
    print(f"PointNet output: {pillar_features.shape}")
    
    # Step 5: Scatter to pseudo-image
    image_height = int((CONFIG['y_range'][1] - CONFIG['y_range'][0]) / CONFIG['grid_size_y'])
    image_width = int((CONFIG['x_range'][1] - CONFIG['x_range'][0]) / CONFIG['grid_size_x'])
    
    coords = torch.from_numpy(pillar_coords[:filled_pillars, :2]).long()
    features = pillar_features[:, :filled_pillars]
    
    pseudo_image = scatter_to_pseudo_image(features, coords, image_height, image_width)
    print(f"Pseudo-image: {pseudo_image.shape}")
    
    # Verify final output
    expected = (CONFIG['feature_channels'], image_height, image_width)
    status = "✅" if pseudo_image.shape == expected else "❌"
    print(f"{status} Final shape: {pseudo_image.shape} (expected: {expected})")
    
    return {
        'pseudo_image': pseudo_image,
        'config': CONFIG,
        'stats': {
            'original_points': len(points),
            'filtered_points': len(filtered_points), 
            'pillars': len(pillars),
            'filled_pillars': filled_pillars
        }
    }


if __name__ == "__main__":
    results = main()
    print("\n=== Pipeline Complete ===")
    print("Ready for 2D CNN Backbone!")