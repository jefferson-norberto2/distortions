import torch
from torch import nn
from distortions.model.extractor import Extractor

class DualStreamV2(nn.Module):
    def __init__(self, model1, model2, num_classes):
        super().__init__()
        self.rgb_arm = Extractor(model1)
        self.hsv_arm = Extractor(model2)

        dim_total = self.rgb_arm.feature_dim + self.hsv_arm.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(dim_total, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, img_rgb, img_hsv):
        feat_rgb = self.rgb_arm(img_rgb)
        feat_hsv = self.hsv_arm(img_hsv)

        features = torch.cat((feat_rgb, feat_hsv), dim=1)
        return self.classifier(features)