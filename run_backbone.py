import torch 
from backbone.backbobe import Backbone
from backbone.backbobe import Neck


# Test the full pipeline
def test_backbone_neck():
    # Create modules
    backbone = Backbone()
    neck = Neck()
    
    # Test input
    x = torch.randn(2, 64, 500, 440)
    
    # Forward pass
    backbone_features = backbone(x)
    print("Backbone outputs:")
    for i, feat in enumerate(backbone_features):
        print(f"  Block {i+1}: {feat.shape}")
    
    # Neck forward
    output = neck(backbone_features)
    print(f"\nNeck output: {output.shape}")
    
    # Verify dimensions
    assert output.shape == torch.Size([2, 384, 250, 220])
    print("✅ All tests passed!")

test_backbone_neck()
