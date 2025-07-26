# test_hybrid_fold6.py

import os
from hybrid_dataset import HybridDataset
from hybrid_model import HybridFusionModel
from retfound.models_vit import create_model

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

# === CONFIG ===
csv_path = "D:/EXPERIMENT/48-HYBRID/TRIAL-RUN/fold_0_right_eye_only_flipped.csv"
img_dir = "D:/EXPERIMENT/48-HYBRID/TRIAL-RUN/images_flipped"
tabular_cols = ["Age_at_baseline", "Sex_0F_1M", "Type1_Diabetes", "Type2_Diabetes", "Hypertension"]
batch_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD DATA ===
df = pd.read_csv(csv_path)
dataset = HybridDataset(df, img_dir=img_dir, tabular_cols=tabular_cols, label_col="MACE_Label")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# === MODEL ===
vit = create_model("RETFound_mae", pretrained=False, num_classes=0, global_pool=True)
model = HybridFusionModel(vit_model=vit, tabular_dim=len(tabular_cols), num_classes=2)
model.to(device)
model.eval()

# === RUN INFERENCE ===
y_true, y_prob = [], []
with torch.no_grad():
    for (img, tab), label in dataloader:
        img, tab = img.to(device), tab.to(device)
        out = model(img, tab)
        prob = torch.softmax(out, dim=1)[:, 1]  # prob for class 1
        y_prob.extend(prob.cpu().numpy())
        y_true.extend(label.numpy())

# === METRICS ===
auc = roc_auc_score(y_true, y_prob)
print(f"✅ AUC on fold_6_test.csv: {auc:.4f}")
