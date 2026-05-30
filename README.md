# 🧠 Alzheimer's Disease MRI Detection

**Detection and Diagnosis of Alzheimer's Disease in Early Stages using a Hybrid Approach**

A deep learning project that classifies brain MRI scans into 4 Alzheimer's disease stages using a **CNN + BiGRU + Attention** hybrid model.

---

## 📊 Overview

| Item | Details |
|------|---------|
| **Model** | CNN (4 blocks) + Bidirectional GRU (2 layers) + Attention Pooling |
| **Dataset** | [Falah/Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI) (HuggingFace) |
| **Training Data** | 5,120 brain MRI images |
| **Test Data** | 1,280 brain MRI images |
| **Input Size** | 128 × 128 grayscale |
| **Classes** | Mild Demented, Moderate Demented, Non Demented, Very Mild Demented |
| **Framework** | PyTorch |

---

## 🏗️ Architecture

```
Input MRI (1 × 128 × 128)
    │
    ├─► CNN Backbone (4 conv blocks with BatchNorm)
    │     → Feature maps (256 × 8 × 8)
    │
    ├─► Reshape to spatial tokens
    │     → Sequence (64 × 256)
    │
    ├─► Bidirectional GRU (2 layers, hidden=128)
    │     → Contextual tokens (64 × 256)
    │
    ├─► Attention Pooling (learned weights)
    │     → Single representation (256)
    │
    └─► Classifier Head
          → 4-class prediction
```

The CNN extracts spatial features, which are reshaped into a sequence of tokens. The BiGRU captures spatial dependencies between regions, and the attention mechanism learns to focus on the most diagnostically relevant areas.

---

## 🚀 Quick Start

### 1. Train the Model (Google Colab)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `train_notebook.ipynb` **OR** upload `model.py` and `train.py`
3. Set runtime to **GPU**: `Runtime → Change runtime type → GPU`
4. Run all cells (~15 minutes)
5. Download `checkpoints/best_model.pth`

**Or run locally (if you have a GPU):**
```bash
pip install -r requirements.txt
python train.py
```

### 2. Run the Web Demo

```bash
# Place best_model.pth in the checkpoints/ folder
pip install -r requirements.txt
streamlit run app.py
```

### 3. CLI Prediction

```bash
python predict.py --image path/to/mri_scan.jpg
python predict.py --image path/to/mri_scan.jpg --save-overlay
```

---

## 📁 Project Structure

```
AlzheimerDetection/
├── model.py              # Hybrid CNN+BiGRU+Attention model
├── train.py              # Training pipeline (Colab-ready)
├── train_notebook.ipynb  # Google Colab notebook
├── app.py                # Streamlit web demo
├── gradcam.py            # Grad-CAM visualization
├── predict.py            # CLI prediction tool
├── requirements.txt      # Dependencies
├── checkpoints/          # Trained model weights
│   └── best_model.pth
├── samples/              # Sample MRI images for demo
└── README.md             # This file
```

---

## 🖥️ Web Demo Features

- **Upload** any brain MRI scan (PNG/JPG)
- **Prediction** with confidence percentage
- **Probability bars** for all 4 classes
- **Grad-CAM heatmap** showing which brain regions the model focuses on
- **Diagnosis summary** with clinical-style recommendations
- **Dark theme** with polished UI

---

## 📋 Training Details

- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler:** Cosine annealing
- **Loss:** Cross-entropy with class weights (handles imbalance)
- **Augmentation:** Random crop, flip, rotation, affine transforms
- **Early Stopping:** Patience = 5 epochs
- **Mixed Precision:** Automatic on GPU

---

## ⚠️ Disclaimer

This is an AI-assisted screening tool for **educational purposes only**. It is NOT a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for clinical decisions.
