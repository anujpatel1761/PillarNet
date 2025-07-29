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

    def forward(self, x):
        """
        Forward pass with detailed shape tracking
        """
        # === INPUT HANDLING ===
        # Input: (D, P, N) = (9, 12000, 100) for single sample
        #    OR: (B, D, P, N) for batched input
        squeeze = False
        if x.dim() == 3:
            # Input: (D, P, N) = (9, 12000, 100)
            x = x.unsqueeze(0)  # Add batch dimension
            # Output: (B, D, P, N) = (1, 9, 12000, 100)
            squeeze = True

        B, D, P, N = x.shape
        # Current shape: (B, D, P, N) = (1, 9, 12000, 100)

        # === STEP 1: FLATTEN FOR CONV1D ===
        x = x.flatten(2)  # Flatten from dimension 2 onwards: P*N
        # Input:  (B, D, P, N) = (1, 9, 12000, 100)
        # Output: (B, D, P*N) = (1, 9, 1200000)

        # === STEP 2: APPLY LINEAR LAYER (1x1 CONVOLUTION) ===
        x = self.conv(x)
        # Input:  (B, D, P*N) = (1, 9, 1200000)
        # Output: (B, C, P*N) = (1, 64, 1200000)

        # === STEP 3: BATCH NORMALIZATION ===
        x = self.bn(x)
        # Input:  (B, C, P*N) = (1, 64, 1200000)
        # Output: (B, C, P*N) = (1, 64, 1200000)

        # === STEP 4: RELU ACTIVATION ===
        x = F.relu(x, inplace=True)
        # Input:  (B, C, P*N) = (1, 64, 1200000)
        # Output: (B, C, P*N) = (1, 64, 1200000)

        # Note: Steps 2-4 can be combined as:
        # x = F.relu(self.bn(self.conv(x)), True)

        # === STEP 5: RESHAPE BACK TO PILLAR FORMAT ===
        x = x.view(B, -1, P, N)  # -1 infers the C dimension (64)
        # Input:  (B, C, P*N) = (1, 64, 1200000)
        # Output: (B, C, P, N) = (1, 64, 12000, 100)

        # === STEP 6: MAX POOLING OVER POINTS ===
        x = x.max(dim=3).values  # Max over N (points dimension)
        # Input:  (B, C, P, N) = (1, 64, 12000, 100)
        # Output: (B, C, P) = (1, 64, 12000)

        # === OUTPUT HANDLING ===
        if squeeze:
            x = x.squeeze(0)  # Remove batch dimension if it was added
            # Input:  (B, C, P) = (1, 64, 12000)
            # Output: (C, P) = (64, 12000)

        return x
        # Final output: (C, P) = (64, 12000) for single sample
        #           OR: (B, C, P) for batched input


def scatter_to_pseudo_image(features, coordinates, H, W):
    """
    Scatter pillar features back to spatial locations to create pseudo-image
    
    Args:
        features: (C, P) = (64, 12000) - learned features for each pillar
        coordinates: (P, 2) = (12000, 2) - [x, y] grid coordinates for each pillar
        H: int - height of pseudo-image canvas (500)
        W: int - width of pseudo-image canvas (440)
    
    Returns:
        pseudo_image: (C, H, W) = (64, 500, 440)
    """
    # === INPUT SHAPES ===
    C, P = features.shape  # (64, 12000)
    # coordinates shape: (P, 2) = (12000, 2)
    
    # === INITIALIZE EMPTY PSEUDO-IMAGE ===
    pseudo_image = torch.zeros(C, H, W, device=features.device)
    # Shape: (C, H, W) = (64, 500, 440)
    
    # === SCATTER FEATURES TO SPATIAL LOCATIONS ===
    for i in range(P):  # Loop through all pillars
        x, y = coordinates[i]  # Get x, y coordinates for pillar i
        # x, y are grid indices
        
        # Bounds checking
        if 0 <= x < W and 0 <= y < H:
            # Place all C features for pillar i at location (y, x)
            pseudo_image[:, y, x] = features[:, i]
            # features[:, i] shape: (C,) = (64,)
            # pseudo_image[:, y, x] shape: (C,) = (64,)
    
    return pseudo_image  # (C, H, W) = (64, 500, 440)
