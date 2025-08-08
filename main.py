"""
main_pipeline.py
"""
import torch
import numpy as np
from configs.pointpillars_kitti import PointPillarsConfig
from preprocess.point_cloud_loader import load_lidar_file, filter_point_cloud_range
from preprocess.pillar_creator import point_cloud_to_pillars  
from preprocess.dense_tensor_creator import create_dense_tensor
from preprocess.pointnet_encoder import PointNetEncoder, scatter_to_pseudo_image_efficient


def main(config_class=PointPillarsConfig):
    cfg = config_class()
    
    print("=== PointPillars Pipeline ===")
    print(f"Config: {config_class.__name__}")
    
    # Step 1: Load and filter point cloud
    file_path = f"{cfg.data_root}/000000.bin"
    points = load_lidar_file(file_path)
    filtered_points = filter_point_cloud_range(points, config=cfg)
    print(f"Points: {len(points)} → {len(filtered_points)} (filtered)")

    # Step 2: Convert to pillars
    pillars, _ = point_cloud_to_pillars(filtered_points, config=cfg)
    print(f"Pillars: {len(pillars)} non-empty")
    
    # Step 3: Create dense tensor - FIX HERE!
    dense_tensor, pillar_coords, filled_pillars, pillar_mask = create_dense_tensor(
        pillars, config=cfg  # Pass config object
    )
    
    # Step 4: PointNet encoding
    encoder = PointNetEncoder(in_channels=9, out_channels=cfg.feature_channels)
    dense_tensor = torch.from_numpy(dense_tensor).float()
    
    with torch.no_grad():
        pillar_features = encoder(dense_tensor, pillar_mask)
    
    # Step 5: Scatter to pseudo-image
    coords = torch.from_numpy(pillar_coords[:filled_pillars]).long()
    features = pillar_features[:, :filled_pillars]
    
    pseudo_image = scatter_to_pseudo_image_efficient(
        features, coords, filled_pillars, cfg.image_height, cfg.image_width
    )
    
    print(f"Pseudo-image: {pseudo_image.shape}")
    print(f"✅ Complete! Shape: {pseudo_image.shape}")
    
    return pseudo_image, cfg


if __name__ == "__main__":
    pseudo_image, config = main()