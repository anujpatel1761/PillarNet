import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class Backbone(nn.Module):
    """
    Top-Down Backbone Network for PointPillars
    Progressively reduces spatial resolution while increasing channels
    """
    
    def __init__(self, in_channel=64, out_channels=[64, 128, 256], layer_nums=[3, 6, 6], layer_strides=[2, 2, 2]):
        """
        Args:
            in_channel: Input channels from pseudo-image (e.g., 64)
            out_channels: List of output channels for each block (e.g., [64, 128, 256])
            layer_nums: List of total conv layers per block (e.g., [3, 6, 6])
            layer_strides: List of downsampling strides per block (e.g., [2, 2, 2])
        
        Example config:
            in_channel=64, out_channels=[64,128,256], layer_nums=[3,6,6], layer_strides=[2,2,2]
            Creates 3 blocks:
            - Block 1: 64→64 channels, 3 convs total, downsample by 2
            - Block 2: 64→128 channels, 6 convs total, downsample by 2  
            - Block 3: 128→256 channels, 6 convs total, downsample by 2
        """
        super().__init__()

        # Ensure all lists have same length (one per block)
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)
        
        # Will store all blocks (Block 1, Block 2, Block 3)
        self.multi_blocks = nn.ModuleList()
        
        # Build each block
        for i in range(len(layer_strides)):
            # Container for all layers in this block
            blocks = []
            
            # ============ FIRST CONVOLUTION (Downsampling) ============
            # This conv:
            # - Changes channels: in_channel → out_channels[i]
            # - Downsamples: stride=layer_strides[i] (typically 2)
            # - Example Block 1: [64,500,440] → [64,250,220]
            blocks.append(nn.Conv2d(
                in_channel,           # Input channels (64 for block 1)
                out_channels[i],      # Output channels (64/128/256)
                kernel_size=3,        # 3×3 convolution
                stride=layer_strides[i],  # Stride 2 = downsample by 2
                bias=False,           # No bias (BatchNorm handles it)
                padding=1             # Padding to maintain size calculation
            ))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))
            
            # ============ REMAINING CONVOLUTIONS (Feature Extraction) ============
            # These convs:
            # - Keep same channels: out_channels[i] → out_channels[i]
            # - No downsampling: stride=1
            # - Extract progressively complex features
            # - Number of these: layer_nums[i] - 1 (because first conv already added)
            
            for j in range(layer_nums[i] - 1):  
                # Example: if layer_nums[i]=3, this loop runs 2 times
                # So total convs = 1 (first) + 2 (loop) = 3
                
                blocks.append(nn.Conv2d(
                    out_channels[i],   # Input = Output channels (stay same)
                    out_channels[i],   # No channel change
                    kernel_size=3,     # 3×3 convolution
                    stride=1,          # No downsampling
                    bias=False,        # No bias needed
                    padding=1          # Maintain spatial size
                ))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))
            # Update in_channel for next block
            # Block 1 outputs 64 channels → Block 2 needs in_channel=64
            # Block 2 outputs 128 channels → Block 3 needs in_channel=128
            in_channel = out_channels[i]
            
            # Combine all layers of this block into a Sequential module
            self.multi_blocks.append(nn.Sequential(*blocks))
        print(f"Final Check how it is print",self.multi_blocks )
        # ============ WEIGHT INITIALIZATION ============
        # Kaiming/He initialization is good for ReLU networks
        # Helps with gradient flow at the start of training
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        """
        Forward pass through all blocks
        
        Args:
            x: Input pseudo-image tensor
               Shape: (batch, channels, height, width)
               Example: (6, 64, 500, 440)
        
        Returns:
            List of feature maps from each block
            Example: [
                (6, 64, 250, 220),   # After Block 1 (stride 2)
                (6, 128, 125, 110),  # After Block 2 (stride 4 total)
                (6, 256, 63, 55)     # After Block 3 (stride 8 total)
            ]
        """
        outs = []  # Store output from each block
        
        # Pass through each block sequentially
        for i in range(len(self.multi_blocks)):
            # Block i processes x and updates it
            x = self.multi_blocks[i](x)
            
            # Save this block's output (needed for upsampling later)
            outs.append(x)
            
            # x becomes input for next block
        
        print("My Out is:",outs)
        return outs
    

class Neck(nn.Module):
    def __init__(self, in_channels=[64, 128, 256], 
                 upsample_strides=[1, 2, 4], 
                 out_channels=[128, 128, 128],
                 use_conv_for_no_stride=True):
        """
        Production-ready implementation matching paper specifications
        """
        super().__init__()
        
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)
        
        self.decoder_blocks = nn.ModuleList()
        self.upsample_strides = upsample_strides
        
        for i in range(len(in_channels)):
            if upsample_strides[i] == 1:
                if use_conv_for_no_stride:
                    # Option 1: Use 1x1 conv (more flexible)
                    up_block = nn.Sequential(
                        nn.Conv2d(in_channels[i], out_channels[i], 1, bias=False),
                        nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True)
                    )
                else:
                    # Option 2: Use identity if channels match
                    if in_channels[i] != out_channels[i]:
                        up_block = nn.Sequential(
                            nn.Conv2d(in_channels[i], out_channels[i], 1, bias=False),
                            nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                            nn.ReLU(inplace=True)
                        )
                    else:
                        up_block = nn.Identity()
            else:
                # Transposed convolution with proper padding
                # Calculate padding to ensure exact dimension match
                up_block = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels[i],
                        out_channels[i],
                        kernel_size=3,  # Fixed kernel size
                        stride=upsample_strides[i],
                        padding=1,
                        output_padding=upsample_strides[i] - 1,  # Ensures proper size
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True)
                )
            
            self.decoder_blocks.append(up_block)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        """
        Combines features from different scales
        Handles dimension mismatches gracefully
        """
        outs = []
        target_size = x[0].shape[2:]  # Use first feature map size as reference
        
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])
            
            # Handle any dimension mismatch due to integer division
            if xi.shape[2:] != target_size:
                # Use bilinear interpolation to match exact size
                xi = F.interpolate(xi, size=target_size, 
                                  mode='bilinear', align_corners=False)
            
            outs.append(xi)
        
        # Concatenate all upsampled features
        out = torch.cat(outs, dim=1)
        return out