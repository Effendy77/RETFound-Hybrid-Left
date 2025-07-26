import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, img_encoder, tabular_dim, hidden_dim=128, dropout=0.3):
        super(HybridModel, self).__init__()
        self.img_encoder = img_encoder

        # RETFound ViT outputs 768-dim for base, 1024/1280 for large/huge
        self.img_feature_dim = self._infer_image_feature_dim()

        # Combine image and tabular features
        self.fc = nn.Sequential(
            nn.Linear(self.img_feature_dim + tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # For binary classification
        )

    def _infer_image_feature_dim(self):
        # Dummy input to infer ViT encoder output size
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            features = self.img_encoder(dummy_input)
        return features.shape[1]

    def forward(self, image, tabular):
        img_feat = self.img_encoder(image)
        x = torch.cat([img_feat, tabular], dim=1)
        return self.fc(x).squeeze(1)  # output shape: (batch,)
