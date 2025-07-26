# main_finetune_hybrid_cv_v3.py
# Full 5-fold CV with early stopping, dynamic class weights, TTA support, and TensorBoard logging

import os
import time
import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from hybrid_dataset import HybridDataset
from hybrid_model import HybridModel
from retfound.models_vit import RETFound_mae

# === CONFIG ===
root = "/home/fendy77/hybrid_project"
fold_dir = os.path.join(root, "folds_trial")
img_dir = os.path.join(root, "images_flipped")
tabular_cols = ['Age_at_baseline', 'Sex_0F_1M', 'diabetes_prevalent', 'hypertension_prevalent']
pretrained_path = os.path.join(root, "RETFound-HYBRID/pretrained/RETFound_cfp_weights.pth")
log_dir = "./tb_logs_hybrid_cv_v3"
os.makedirs(log_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === CROSS-VALIDATION LOOP ===
for k in range(5):
    print(f"\n=== Fold {k} ===")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"fold{k}"))

    train_dfs = [pd.read_csv(os.path.join(fold_dir, f"fold_{i}_trial.csv")) for i in range(5) if i != k]
    val_df = pd.read_csv(os.path.join(fold_dir, f"fold_{k}_trial.csv"))
    train_df = pd.concat(train_dfs).reset_index(drop=True)

    train_dataset = HybridDataset(train_df, img_dir, tabular_cols, transform=transform)
    val_dataset = HybridDataset(val_df, img_dir, tabular_cols, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    vit = RETFound_mae()
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder_") and "head" not in k}
    vit.load_state_dict(filtered_state_dict, strict=False)

    model = HybridModel(vit, tabular_dim=len(tabular_cols)).to(device)

    mace_counts = train_df['MACE_Label'].value_counts().to_dict()
    num_class_0 = mace_counts.get(0, 1)
    num_class_1 = mace_counts.get(1, 1)
    total = num_class_0 + num_class_1
    weight_0 = total / (2 * num_class_0)
    weight_1 = total / (2 * num_class_1)
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)
    print(f"Class Weights: [No-MACE={weight_0:.4f}, MACE={weight_1:.4f}]")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_auc = 0.0
    best_epoch = 0
    patience = 10
    patience_counter = 0

    for epoch in range(1, 50):
        model.train()
        total_loss = 0
        for img, tab, label in train_loader:
            img, tab, label = img.to(device), tab.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(img, tab)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for img, tab, label in val_loader:
                img, tab, label = img.to(device), tab.to(device), label.to(device)
                out = model(img, tab)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(out, dim=1).cpu().numpy()
                labels = label.cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels)

        auc = roc_auc_score(all_labels, all_probs)
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}, Val AUC: {auc:.4f}")
        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("AUC/val", auc, epoch)

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), f"checkpoints/checkpoint-fold{k}.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    print(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")

    pd.DataFrame({
        "fold": [k],
        "best_auc": [best_auc],
        "best_epoch": [best_epoch],
        "val_samples": [len(val_df)]
    }).to_csv(f"results_cv_fold_{k}.csv", index=False)

    writer.close()

print("✅ Completed 5-Fold CV with TensorBoard and dynamic class weighting")