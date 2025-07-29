import numpy as np
import random

def create_dense_tensor(augmented_pillars, max_pillars=12000, max_points_per_pillar=100):
    """
    Create dense tensor from augmented pillars with sparsity handling.
    
    This function implements the dense tensor creation from PointPillars paper:
    1. Limit number of non-empty pillars per sample (P)
    2. Limit number of points per pillar (N) 
    3. Apply random sampling when limits exceeded
    4. Apply zero padding when insufficient data
    5. Create final dense tensor of size (D, P, N) = (9, P, N)
    
    Args:
        augmented_pillars (dict): Dictionary with pillar_id as key and 9D points as value
        max_pillars (int): Maximum number of pillars per sample (P)
        max_points_per_pillar (int): Maximum number of points per pillar (N)
    
    Returns:
        np.ndarray: Dense tensor of shape (max_pillars, max_points_per_pillar, 9)
        np.ndarray: Pillar coordinates of shape (max_pillars, 3) for scattering back
        int: Number of actual filled pillars
    """
    
    # Step 1: Handle pillar sampling if too many pillars
    pillar_ids = list(augmented_pillars.keys())
    
    if len(pillar_ids) > max_pillars:
        print(f"Too many pillars ({len(pillar_ids)}), sampling {max_pillars}")
        selected_pillar_ids = random.sample(pillar_ids, max_pillars) 
    else:
        selected_pillar_ids = pillar_ids
    
    # Step 2: Initialize dense tensor with zeros (automatic padding)
    # Shape: (P, N, D) where D=9 dimensions
    dense_tensor = np.zeros((max_pillars, max_points_per_pillar, 9), dtype=np.float32)
    
    # Step 3: Initialize pillar coordinates for scattering back to pseudo-image
    # Format: [batch_index, y_coord, x_coord] (batch=0 for single sample)
    pillar_coordinates = np.zeros((max_pillars, 3), dtype=np.int32)
    
    # Step 4: Fill dense tensor with actual pillar data
    filled_pillars = 0
    
    for i, pillar_id in enumerate(selected_pillar_ids):
        points_in_pillar = augmented_pillars[pillar_id]
        
        # Handle point sampling if too many points per pillar
        if len(points_in_pillar) > max_points_per_pillar:
            sampled_points = random.sample(points_in_pillar, max_points_per_pillar)
        else:
            sampled_points = points_in_pillar
        
        # Fill dense tensor with sampled points
        for j, point in enumerate(sampled_points):
            dense_tensor[i, j, :] = point
        
        # Store pillar coordinates for scattering back to pseudo-image
        # pillar_id = (pillar_x, pillar_y)
        pillar_coordinates[i] = [0, pillar_id[1], pillar_id[0]]  # [batch, y, x]
        
        filled_pillars += 1
        
    
    # Step 5: Print statistics
    empty_pillars = max_pillars - filled_pillars
    sparsity = (empty_pillars / max_pillars) * 100
    
    print(f"\n=== DENSE TENSOR STATISTICS ===")
    print(f"Total pillars capacity: {max_pillars}")
    print(f"Filled pillars: {filled_pillars}")
    print(f"Empty pillars (zero-padded): {empty_pillars}")
    print(f"Sparsity: {sparsity:.1f}%")
    print(f"Points per pillar limit: {max_points_per_pillar}")
    print(f"Dense tensor shape: {dense_tensor.shape}")
    print(f"Pillar coordinates shape: {pillar_coordinates.shape}")
    
    return dense_tensor, pillar_coordinates, filled_pillars

def create_pseudo_image_indices(pillar_coordinates, filled_pillars, image_height, image_width):
    """
    Create indices for scattering pillar features back to pseudo-image format.
    
    Args:
        pillar_coordinates (np.ndarray): Pillar coordinates of shape (P, 3)
        filled_pillars (int): Number of actual filled pillars
        image_height (int): Height of pseudo-image (typically H = y_range / grid_size)  
        image_width (int): Width of pseudo-image (typically W = x_range / grid_size)
    
    Returns:
        np.ndarray: Indices for scattering features back to (C, H, W) format
    """
    # Only use coordinates for filled pillars
    valid_coords = pillar_coordinates[:filled_pillars]
    
    # Extract batch, y, x indices
    batch_indices = valid_coords[:, 0]  # Always 0 for single sample
    y_indices = valid_coords[:, 1] 
    x_indices = valid_coords[:, 2]
    
    print(f"Pseudo-image size: ({image_height}, {image_width})")
    print(f"Valid pillar coordinates: {filled_pillars}")
    
    return batch_indices, y_indices, x_indices