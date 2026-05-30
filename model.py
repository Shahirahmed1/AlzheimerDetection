"""
Alzheimer's MRI Classification — ResNet18 + BiGRU + Attention Hybrid

Dual-head architecture:
    1. ResNet18 backbone (pretrained, adapted to 1-channel grayscale)
    2. Head A: Global Average Pooling → FC classifier (stable baseline)
    3. Head B: Spatial tokens → BiGRU → Attention → classifier
    4. Fusion: 0.6 * Head_A + 0.4 * Head_B (proven effective)

Classes: Mild_Demented, Moderate_Demented, Non_Demented, Very_Mild_Demented
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


CLASS_NAMES = ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 128


class AttentionPooling(nn.Module):
    """Learned attention weights over sequence tokens."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1)
        )
    def forward(self, x):
        return (F.softmax(self.attn(x), dim=1) * x).sum(dim=1)


class AlzheimerHybridModel(nn.Module):
    """
    ResNet18 + BiGRU + Attention Hybrid with dual-head fusion.

    Head A (CNN path):   backbone → global_avg_pool → FC → logits_A
    Head B (GRU path):   backbone → spatial_tokens → BiGRU → attention → FC → logits_B
    Output:              0.6 * logits_A + 0.4 * logits_B
    """

    def __init__(self, num_classes=NUM_CLASSES, gru_hidden=256, dropout=0.3, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        # ── ResNet18 Backbone ──
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Adapt conv1: RGB(3ch) → Grayscale(1ch)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        feat_dim = 512  # ResNet18 output channels

        # ── Head A: Global Avg Pool → FC (stable CNN classifier) ──
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_a = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        # ── Head B: BiGRU + Attention ──
        self.gru = nn.GRU(
            input_size=feat_dim, hidden_size=gru_hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        gru_out_dim = gru_hidden * 2
        self.attention = AttentionPooling(gru_out_dim)
        self.head_b = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        # Fusion weights
        self.cnn_weight = 0.6
        self.gru_weight = 0.4

    def _backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        features = self._backbone(x)          # (B, 512, 4, 4)

        # Head A: CNN classifier
        pooled = self.global_pool(features)    # (B, 512, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)
        logits_a = self.head_a(pooled)

        # Head B: GRU + Attention
        B, C, H, W = features.shape
        tokens = features.view(B, C, H*W).permute(0, 2, 1).contiguous()  # (B, 16, 512)
        gru_out, _ = self.gru(tokens)
        attended = self.attention(gru_out)
        logits_b = self.head_b(attended)

        # Weighted fusion
        return self.cnn_weight * logits_a + self.gru_weight * logits_b

    def get_last_conv_layer(self):
        """For GradCAM — returns last conv in ResNet backbone."""
        return self.layer4[-1].conv2


def build_model(num_classes=NUM_CLASSES, pretrained=True, device="cpu"):
    return AlzheimerHybridModel(num_classes=num_classes, pretrained=pretrained).to(device)


def load_trained_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    else:
        state_dict = checkpoint
        num_classes = NUM_CLASSES
    model = AlzheimerHybridModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


if __name__ == "__main__":
    m = build_model(pretrained=False)
    print(f"Params: {sum(p.numel() for p in m.parameters()):,}")
    o = m(torch.randn(2, 1, 128, 128))
    print(f"Output: {o.shape}")
    print("✅ OK")
