"""
Training Script — ResNet18 + BiGRU + Attention Hybrid (v3)

Key changes from v2:
  - NO class weights (caused class collapse)
  - NO label smoothing
  - Dual-head fusion (CNN + GRU, 0.6/0.4 weighted)
  - Plain CrossEntropyLoss
  - Adam lr=3e-4
  - Simple augmentation
  - Gradient clip 5.0

Usage: python train.py
"""

import os, time, random, math, json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from model import AlzheimerHybridModel, CLASS_NAMES, NUM_CLASSES, IMG_SIZE

SEED = 42; BATCH_SIZE = 32; NUM_EPOCHS = 25; LR = 3e-4; PATIENCE = 6
CHECKPOINT_DIR = Path("checkpoints"); CHECKPOINT_DIR.mkdir(exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tfm = transforms.Compose([
    transforms.Resize((140, 140)), transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10),
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
])
val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
])

class MRIDataset(Dataset):
    def __init__(self, hf_ds, tfm):
        self.ds = hf_ds; self.tfm = tfm
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        ex = self.ds[idx]; img = ex['image']
        if not isinstance(img, Image.Image):
            arr = np.squeeze(np.array(img))
            if arr.dtype != np.uint8: arr = (255*(arr-arr.min())/(arr.max()-arr.min()+1e-8)).astype(np.uint8)
            img = Image.fromarray(arr, mode='L')
        elif img.mode != 'L': img = img.convert('L')
        return self.tfm(img), int(ex['label'])

def main():
    print("=" * 60)
    print("  Alzheimer's MRI — ResNet18 + BiGRU + Attention (v3)")
    print(f"  Device: {DEVICE} | LR: {LR} | Epochs: {NUM_EPOCHS}")
    print("=" * 60)

    train_ds = load_dataset("Falah/Alzheimer_MRI", split="train")
    test_ds = load_dataset("Falah/Alzheimer_MRI", split="test")
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(MRIDataset(train_ds, train_tfm), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(MRIDataset(test_ds, val_tfm), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = AlzheimerHybridModel(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler() if DEVICE.type == "cuda" else None

    best_acc = 0.0; no_improve = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); rloss = 0; correct = 0; tot = 0; t0 = time.time()
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast(device_type="cuda"):
                    out = model(imgs); loss = criterion(out, labs)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer); scaler.update()
            else:
                out = model(imgs); loss = criterion(out, labs)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            rloss += loss.item()*imgs.size(0); correct += (out.argmax(1)==labs).sum().item(); tot += labs.size(0)
        scheduler.step()

        model.eval(); vc=0; vt=0; vl=0
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                out = model(imgs); loss = criterion(out, labs)
                vl += loss.item()*imgs.size(0); vc += (out.argmax(1)==labs).sum().item(); vt += labs.size(0)
        vacc = vc/vt; mark = ""
        if vacc > best_acc:
            best_acc = vacc; no_improve = 0; mark = " ⭐"
            torch.save({'epoch':epoch, 'model_state_dict':model.state_dict(), 'val_acc':vacc,
                        'num_classes':NUM_CLASSES, 'class_names':CLASS_NAMES}, CHECKPOINT_DIR/"best_model.pth")
        else: no_improve += 1
        print(f"  Epoch {epoch:2d}/{NUM_EPOCHS} │ loss:{rloss/tot:.4f} acc:{correct/tot:.4f} │ val:{vacc:.4f} │ {time.time()-t0:.1f}s{mark}")
        if no_improve >= PATIENCE: print("  ⏹ Early stopping"); break

    ckpt = torch.load(CHECKPOINT_DIR/"best_model.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            all_p.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy()); all_l.extend(labs.numpy())
    print(f"\n🎯 Best: {best_acc*100:.2f}%\n")
    print(classification_report(all_l, all_p, target_names=CLASS_NAMES, zero_division=0))

if __name__ == "__main__":
    main()
