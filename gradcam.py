"""
Grad-CAM Visualization for Alzheimer's MRI Classification

Generates class-activation heatmaps showing which brain regions
the CNN focuses on when making a prediction.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.cm as cm


# ── Standard preprocessing (must match training) ───────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def pil_to_tensor(pil_img: Image.Image, device: str = "cpu") -> torch.Tensor:
    """Convert a PIL image to a preprocessed (1,1,128,128) tensor."""
    img = pil_img.convert("L")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    return tensor


class GradCAM:
    """
    Grad-CAM implementation for any CNN-based model.

    Usage:
        cam = GradCAM(model, target_layer)
        heatmap, pred_class, probs = cam(pil_image, device="cpu")
        overlay = cam.overlay_heatmap(pil_image, heatmap, alpha=0.45)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._activations = None
        self._gradients = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self._activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        pil_img: Image.Image,
        target_class: int = None,
        device: str = "cpu",
    ):
        """
        Compute Grad-CAM heatmap for the given image.

        Args:
            pil_img: Input PIL image (any mode, will be converted to grayscale)
            target_class: Class index to visualize. If None, uses predicted class.
            device: torch device string

        Returns:
            heatmap: (H, W) numpy array in [0, 1]
            pred_class: predicted class index
            probs: (num_classes,) probability array
        """
        self.model.eval()
        tensor = pil_to_tensor(pil_img, device=device)

        # Forward pass
        tensor.requires_grad_(True)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        # Compute Grad-CAM
        activations = self._activations.squeeze(0)  # (C, H, W)
        gradients = self._gradients.squeeze(0)       # (C, H, W)

        # Global average pooling of gradients → channel weights
        weights = gradients.mean(dim=(1, 2))          # (C,)

        # Weighted combination of activation maps
        cam = torch.relu((weights.view(-1, 1, 1) * activations).sum(dim=0))

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap = cam.cpu().numpy()  # (H, W)
        return heatmap, int(target_class), probs

    @staticmethod
    def overlay_heatmap(
        pil_img: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> Image.Image:
        """
        Overlay Grad-CAM heatmap on the original image.

        Args:
            pil_img: Original PIL image
            heatmap: (H, W) numpy array in [0, 1] from __call__
            alpha: Blending factor (0 = original only, 1 = heatmap only)

        Returns:
            Blended PIL image (RGB)
        """
        # Convert heatmap to RGB using jet colormap
        colored = cm.jet(heatmap)[:, :, :3]  # (H, W, 3) float in [0, 1]
        heat_img = Image.fromarray((colored * 255).astype(np.uint8))
        heat_img = heat_img.resize(pil_img.size, resample=Image.BILINEAR)

        # Convert original to RGB
        base = pil_img.convert("RGB")

        # Blend
        overlay = Image.blend(base, heat_img, alpha=alpha)
        return overlay

    def remove_hooks(self):
        """Clean up registered hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


def make_gradcam(model) -> GradCAM:
    """
    Create a GradCAM instance using the model's last Conv2d layer.

    Works with AlzheimerHybridModel (uses get_last_conv_layer method)
    or any model with Conv2d layers.
    """
    if hasattr(model, "get_last_conv_layer"):
        target_layer = model.get_last_conv_layer()
    else:
        # Fallback: find last Conv2d
        target_layer = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        if target_layer is None:
            raise RuntimeError("No Conv2d layer found in model for Grad-CAM")

    return GradCAM(model, target_layer)
