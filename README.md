# RETFound-Hybrid-Left
A hybrid deep learning pipeline combining retinal images (left eye only) and tabular clinical features to predict major adverse cardiovascular events (MACE) in diabetic cohorts. This implementation uses RETFound ViT as the image backbone and supports 5-fold cross-validation with early stopping, dynamic class weights, and final held-out testing.

## 🧠 Key Features
- RETFound ViT-based encoder
- Left-eye only image processing (flipping removed)
- Integration of structured tabular features
- Dynamic class weighting for imbalanced labels
- 5-fold cross-validation with early stopping
- TensorBoard logging and per-fold checkpoints
- Final evaluation on held-out test set (Fold 6)

## 🗂️ Folder Structure
```
RETFound-Hybrid-Left/
├── main_finetune_leftonly.py         # 5-fold CV training script
├── test_fold_6_leftonly.py           # Inference on held-out test set
├── run_all_folds_leftonly.sh         # Shell script to run all folds
├── hybrid_dataset.py                 # Dataset class with image/tabular fusion
├── hybrid_model.py                   # Fusion model architecture
├── retfound/                         # RETFound ViT model
├── folds_MACE_final/                 # 5-fold CSV splits
├── left_fundus_images/               # Image folder (left eye only)
├── checkpoints/                      # Trained model checkpoints
├── tb_logs_hybrid_cv_leftonly/       # TensorBoard logs
├── requirements.txt
├── .gitignore
└── README.md
```

## 📋 Required Input
- CSVs for each fold in `folds_MACE_final/` with columns:
  - `image_filename`, `MACE_Label`, and tabular features
- Left-eye images in `left_fundus_images/` using format: `eid_21015_0_0.png`

## 🚀 How to Train
```bash
bash run_all_folds_leftonly.sh
```

Each fold will save checkpoints to `checkpoints/` and logs to `tb_logs_hybrid_cv_leftonly/`.

## 🧪 How to Test on Fold 6
```bash
python test_fold_6_leftonly.py
```

Ensure your test CSV and image paths are correctly defined inside the script.

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 📍 Notes
- Only **left-eye images (21015)** are used.
- This pipeline **excludes image flipping**.
- Tabular features are customizable via script arguments.

## Data Availability

No UK Biobank participant-level data (including eid) is included in this repository.
All analyses were conducted under UK Biobank approval using secure local data.
