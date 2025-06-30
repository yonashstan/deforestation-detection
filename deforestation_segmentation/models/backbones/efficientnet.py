"""
EfficientNet backbone implementation for forest loss detection.
Optimized for MacBook M1 architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2

class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone with multi-level feature extraction"""
    
    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        pretrained: bool = True,
        features: List[int] = [32, 64, 128, 256]
    ):
        super().__init__()
        
        # Load pretrained EfficientNet
        if model_name == "efficientnet-b0":
            self.model = efficientnet_b0(pretrained=pretrained)
        elif model_name == "efficientnet-b1":
            self.model = efficientnet_b1(pretrained=pretrained)
        elif model_name == "efficientnet-b2":
            self.model = efficientnet_b2(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        # Feature extraction layers
        self.features = nn.ModuleList([
            nn.Sequential(
                self.model.features[0],  # Initial conv
                self.model.features[1],  # First MBConv block
            ),
            self.model.features[2],      # Second MBConv block
            self.model.features[3],      # Third MBConv block
            self.model.features[4],      # Fourth MBConv block
        ])
        
        # Feature projection layers
        self.projections = nn.ModuleList([
            nn.Conv2d(feat, out_feat, 1)
            for feat, out_feat in zip([16, 24, 40, 80], features)
        ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize projection layer weights"""
        for m in self.projections:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-level features"""
        features = []
        
        for feature_layer, projection in zip(self.features, self.projections):
            x = feature_layer(x)
            features.append(projection(x))
        
        return features

class EfficientNetWithAttention(EfficientNetBackbone):
    """EfficientNet backbone with attention mechanism"""
    
    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        pretrained: bool = True,
        features: List[int] = [32, 64, 128, 256],
        attention_heads: int = 4
    ):
        super().__init__(model_name, pretrained, features)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=feat,
                num_heads=attention_heads,
                batch_first=True
            )
            for feat in features
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features with attention"""
        features = super().forward(x)
        
        # Apply attention to each feature level
        attended_features = []
        for feat, attention in zip(features, self.attention_layers):
            # Reshape for attention
            b, c, h, w = feat.shape
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            
            # Apply self-attention
            attn_output, _ = attention(feat_flat, feat_flat, feat_flat)
            
            # Reshape back
            attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)
            attended_features.append(attn_output)
        
        return attended_features 