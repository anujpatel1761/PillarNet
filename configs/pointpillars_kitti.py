"""
PointPillars configuration for KITTI dataset
"""

class PointPillarsConfig:
    # Detection range (meters)
    x_range = (0, 70.4)
    y_range = (-40, 40) 
    z_range = (-3, 1)
    
    # Pillar configuration
    grid_size_x = 0.16
    grid_size_y = 0.16
    max_pillars = 12000
    max_points_per_pillar = 100
    
    # Network architecture
    feature_channels = 64
    backbone_channels = [64, 128, 256]
    neck_upsample_strides = [1, 2, 4]
    neck_out_channels = [128, 128, 128]
    
    # Training parameters
    batch_size = 2
    learning_rate = 0.001
    num_epochs = 80
    
    # Dataset paths
    data_root = "C:/Users/anujp/Desktop/PillarNet/data"
    train_split = "training"
    val_split = "validation"
    
    # Anchor configuration
    anchor_sizes = {
        'car': [3.9, 1.6, 1.56],
        'pedestrian': [0.8, 0.6, 1.73],
        'cyclist': [1.76, 0.6, 1.73]
    }
    anchor_rotations = [0, 1.57]  # 0 and 90 degrees
    
    # Loss weights
    loss_weights = {
        'classification': 1.0,
        'localization': 2.0,
        'direction': 0.2
    }
    
    @property
    def image_height(self):
        return int((self.y_range[1] - self.y_range[0]) / self.grid_size_y)
    
    @property
    def image_width(self):
        return int((self.x_range[1] - self.x_range[0]) / self.grid_size_x)