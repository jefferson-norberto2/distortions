from torch import nn, cat
from distortions.utils.extractor import Extractor
import yaml
from pathlib import Path

class DualStream(nn.Module):
    def __init__(self, rgb_head, hsv_head, num_classes):
        super().__init__()
        self.rgb_head = Extractor(rgb_head)
        self.hsv_head = Extractor(hsv_head)

        # Assuming both extractors return the same dimension (e.g., 2048)
        dim_rgb = self.rgb_head.feature_dim
        dim_hsv = self.hsv_head.feature_dim
        dim_total = dim_rgb + dim_hsv

        # Normalize features independently before merging
        self.norm_rgb = nn.BatchNorm1d(dim_rgb)
        self.norm_hsv = nn.BatchNorm1d(dim_hsv)

        # Gradual reduction
        hidden_dim1 = dim_total // 2

        self.classifier = nn.Sequential(
            nn.Linear(dim_total, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim1, num_classes),
        )

    def forward(self, img_rgb, img_hsv):
        feat_rgb = self.norm_rgb(self.rgb_head(img_rgb))
        feat_hsv = self.norm_hsv(self.hsv_head(img_hsv))

        norm_feat_rgb = self.norm_rgb(feat_rgb)
        norm_feat_hsv = self.norm_hsv(feat_hsv)

        features = cat((norm_feat_rgb, norm_feat_hsv), dim=1)
        return self.classifier(features)
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Factory method to instantiate the model directly from a YAML configuration file.
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        # Open and parse the YAML file safely
        with open(yaml_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        print(f"📄 Loaded config from {yaml_path.name}")

        if not all(key in config for key in ['rgb_head', 'hsv_head', 'num_classes']):
            raise ValueError("YAML config must contain 'rgb_head', 'hsv_head', and 'num_classes' keys.")
        
        return cls(
            rgb_head=config['rgb_head'],
            hsv_head=config['hsv_head'],
            num_classes=config['num_classes']
        )