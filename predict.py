"""
CLI Prediction Script for Alzheimer's MRI Detection

Usage:
    python predict.py --image path/to/mri.jpg
    python predict.py --image path/to/mri.jpg --save-overlay
    python predict.py --image path/to/mri.jpg --checkpoint checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model import load_trained_model, CLASS_NAMES, NUM_CLASSES
from gradcam import make_gradcam, pil_to_tensor


SEVERITY_MAP = {
    "Non_Demented": ("✅ No Dementia", "Routine follow-up recommended."),
    "Very_Mild_Demented": ("⚠️  Very Mild", "Neurologist consultation recommended."),
    "Mild_Demented": ("🟠 Mild", "Specialist referral and cognitive therapy recommended."),
    "Moderate_Demented": ("🔴 Moderate", "Urgent specialist evaluation required."),
}


def predict_image(image_path: str, checkpoint_path: str, save_overlay: bool = False):
    """Run prediction on a single MRI image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    print(f"\n📷 Loading image: {image_path}")
    pil_img = Image.open(image_path).convert("L")
    print(f"   Size: {pil_img.size}")

    # Load model
    print(f"🧠 Loading model: {checkpoint_path}")
    model = load_trained_model(checkpoint_path, device=device)
    print(f"   Device: {device}")

    # Inference
    print("\n🔍 Running inference...")
    tensor = pil_to_tensor(pil_img, device=device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # Display results
    print("\n" + "═" * 50)
    print("  PREDICTION RESULTS")
    print("═" * 50)

    severity_label, recommendation = SEVERITY_MAP[pred_name]
    print(f"\n  Prediction:  {pred_name.replace('_', ' ')}")
    print(f"  Severity:    {severity_label}")
    print(f"  Confidence:  {confidence * 100:.1f}%")
    print(f"\n  📋 {recommendation}")

    print(f"\n  All probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        bar_len = int(probs[i] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        marker = " ◀" if i == pred_idx else ""
        print(f"    {name:25s} {bar} {probs[i]*100:5.1f}%{marker}")

    # Grad-CAM overlay
    if save_overlay:
        print("\n🔥 Generating Grad-CAM overlay...")
        try:
            cam = make_gradcam(model)
            heatmap, _, _ = cam(pil_img, device=device)
            overlay = cam.overlay_heatmap(pil_img, heatmap)
            cam.remove_hooks()

            out_path = Path(image_path).stem + "_gradcam.png"
            overlay.save(out_path)
            print(f"   ✅ Saved overlay to: {out_path}")
        except Exception as e:
            print(f"   ❌ Grad-CAM failed: {e}")

    print("\n" + "═" * 50)
    print("  ⚠️  Disclaimer: AI screening tool only. Consult a doctor.")
    print("═" * 50 + "\n")

    return pred_name, confidence, probs


def main():
    parser = argparse.ArgumentParser(
        description="Alzheimer's MRI Detection — CLI Prediction Tool",
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the brain MRI image file (png/jpg)",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default=str(ROOT / "checkpoints" / "best_model.pth"),
        help="Path to model checkpoint (default: checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--save-overlay", "-s",
        action="store_true",
        help="Save Grad-CAM overlay image",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("   Train the model first: python train.py")
        sys.exit(1)

    predict_image(args.image, args.checkpoint, args.save_overlay)


if __name__ == "__main__":
    main()
