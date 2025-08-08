"""
pointnet_encoder.py

Correctly implemented PointNet encoder with detailed input/output comments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, in_channels=9, out_channels=64):
        super().__init__()
        # Linear layer implemented as 1x1 convolution (as mentioned in paper)
        self.conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # BatchNorm for training stability (bias=False since BN handles bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, pillar_mask=None):
        """
        Forward pass with detailed shape tracking
        """
        # === INPUT HANDLING ===
        squeeze = False
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
            squeeze = True

        B, D, P, N = x.shape

        # === STEP 1: FLATTEN FOR CONV1D ===
        x = x.flatten(2)  # Flatten P*N

        # === STEP 2-4: CONV + BN + RELU ===
        x = F.relu(self.bn(self.conv(x)), inplace=True)

        # === STEP 5: RESHAPE BACK ===
        x = x.view(B, -1, P, N)

        # === STEP 6: MAX POOLING ===
        x = x.max(dim=3).values  # (B, C, P)

        # === STEP 7: APPLY MASK (to zero out empty pillars) ===
        if pillar_mask is not None:
            mask = torch.from_numpy(pillar_mask).to(x.device).float()
            if squeeze:  # Single sample case
                x = x * mask.unsqueeze(0).unsqueeze(1)  # (1, 1, P)
            else:  # Batch case
                x = x * mask.unsqueeze(0).unsqueeze(1)  # Works for both
    
        # # Simpler (also works):
        # if pillar_mask is not None:
        #     mask = torch.from_numpy(pillar_mask).to(x.device).float()
        #     x = x * mask.view(1, 1, -1)  # Works for both cases


        # === OUTPUT HANDLING - MUST KEEP THIS! ===
        if squeeze:
            x = x.squeeze(0)  # CRITICAL: Remove batch dimension
            # (1, 64, 12000) → (64, 12000)

        return x
    

# def scatter_to_pseudo_image(features, coordinates, H, W):
#     """
#     Scatter pillar features back to spatial locations to create pseudo-image
    
#     Args:
#         features: (C, P) = (64, 12000) - learned features for each pillar
#         coordinates: (P, 2) = (12000, 2) - [x, y] grid coordinates for each pillar
#         H: int - height of pseudo-image canvas (500)
#         W: int - width of pseudo-image canvas (440)
    
#     Returns:
#         pseudo_image: (C, H, W) = (64, 500, 440)
#     """
#     # === INPUT SHAPES ===
#     C, P = features.shape  # (64, 12000)
#     # coordinates shape: (P, 2) = (12000, 2)
    
#     # === INITIALIZE EMPTY PSEUDO-IMAGE ===
#     pseudo_image = torch.zeros(C, H, W, device=features.device)
#     # Shape: (C, H, W) = (64, 500, 440)
    
#     # === SCATTER FEATURES TO SPATIAL LOCATIONS ===
#     for i in range(P):  # Loop through all pillars
#         x, y = coordinates[i]  # Get x, y coordinates for pillar i
#         # x, y are grid indices
        
#         # Bounds checking
#         if 0 <= x < W and 0 <= y < H:
#             # Place all C features for pillar i at location (y, x)
#             pseudo_image[:, y, x] = features[:, i]
#             # features[:, i] shape: (C,) = (64,)
#             # pseudo_image[:, y, x] shape: (C,) = (64,)
    
#     return pseudo_image  # (C, H, W) = (64, 500, 440)

def scatter_to_pseudo_image_efficient(features, coordinates, filled_pillars, H, W):
    """
    Efficient scatter using PyTorch advanced indexing
    
    Args:
        features: (C, P) - learned features for each pillar
        coordinates: (P, 2) - [x, y] grid coordinates for each pillar
        filled_pillars: int - number of actually filled pillars
        H, W: canvas dimensions
    
    Returns:
        pseudo_image: (C, H, W)
    """
    C, P = features.shape
    device = features.device
    
    # Initialize pseudo-image
    pseudo_image = torch.zeros(C, H, W, device=device)
    
    # Only use filled pillars (rest are zero-padded)
    valid_coords = coordinates[:filled_pillars]
    valid_features = features[:, :filled_pillars]
    
    # Extract x, y indices (coordinates are [x, y])
    x_indices = valid_coords[:, 0].long()
    y_indices = valid_coords[:, 1].long()
    
    # Bounds checking
    valid = (x_indices >= 0) & (x_indices < W) & (y_indices >= 0) & (y_indices < H)
    
    # Scatter using advanced indexing
    pseudo_image[:, y_indices[valid], x_indices[valid]] = valid_features[:, valid]
    
    return pseudo_image
