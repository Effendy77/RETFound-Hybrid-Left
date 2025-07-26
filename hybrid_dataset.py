import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class HybridDataset(Dataset):
    def __init__(self, csv_path, img_dir, tabular_cols, target_col='MACE_Label', transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.tabular_cols = tabular_cols
        self.target_col = target_col
        self.transform = transform

        # Drop rows with missing image or label
        self.df = self.df[self.df['image_filename'].notna()]
        self.df = self.df[self.df[self.target_col].notna()]
        self.df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.img_dir, row['image_filename'])
        image = Image.open(img_path).convert('RGB')

        # Flip if right eye used
        if 'flip' in row and row['flip'] == 1:
            image = TF.hflip(image)

        if self.transform:
            image = self.transform(image)

        # Tabular features
        tabular_data = torch.tensor(row[self.tabular_cols].values, dtype=torch.float)

        # Target
        label = torch.tensor(row[self.target_col], dtype=torch.float)

        return image, tabular_data, label
