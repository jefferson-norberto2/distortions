from torch import nn, cat
from distortions.utils.extractor import Extractor

class DualStream(nn.Module):
    def __init__(self, model1, model2, num_classes):
        super().__init__()
        self.rgb_head = Extractor(model1)
        self.hsv_head = Extractor(model2)

        # Assuming both extractors return the same dimension (e.g., 2048)
        dim_rgb = self.rgb_head.feature_dim
        dim_hsv = self.hsv_head.feature_dim
        dim_total = dim_rgb + dim_hsv

        # Normalize features independently before merging
        self.norm_rgb = nn.BatchNorm1d(dim_rgb)
        self.norm_hsv = nn.BatchNorm1d(dim_hsv)

        # Gradual reduction
        hidden_dim1 = dim_total // 2
        hidden_dim2 = hidden_dim1 // 2
        hidden_dim3 = hidden_dim2 // 2

        self.classifier = nn.Sequential(
            nn.Linear(dim_total, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim3, num_classes)
        )

    def forward(self, img_rgb, img_hsv):
        feat_rgb = self.norm_rgb(self.rgb_head(img_rgb))
        feat_hsv = self.norm_hsv(self.hsv_head(img_hsv))

        features = cat((feat_rgb, feat_hsv), dim=1)
        return self.classifier(features)