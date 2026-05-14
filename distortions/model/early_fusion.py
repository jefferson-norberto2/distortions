import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class EarlyFusionAdapter(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. The Channel Mixer (Early Fusion)
        # Takes 6 channels (3 RGB + 3 HSV) and projects them into 3 channels
        self.channel_mixer = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)
        self.mixer_activation = nn.LeakyReLU(0.1)
        
        # 2. The Main Backbone
        # CRITICAL FIX: num_classes=0 strips the 1000-class ImageNet head
        # and forces the model to return the raw 1024 feature vector.
        self.backbone = timm.create_model(
            'mobilenetv1_100', 
            pretrained=True, 
            num_classes=0
        )
        
        # 3. Final Classifier
        feature_dim = self.backbone.num_features # This will correctly be 1024
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, img_rgb, img_hsv):
        # Step 1: Pixel-level concatenation along the channel dimension
        x = torch.cat((img_rgb, img_hsv), dim=1)
        
        # Step 2: Channel Mixing
        x = self.channel_mixer(x)
        x = self.mixer_activation(x)
        
        # Step 3: Spatial Resizing
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Step 4: Extract features through the backbone
        # Now this will correctly output a tensor of shape (Batch, 1024)
        features = self.backbone(x)
        
        # Step 5: Classification
        return self.classifier(features)