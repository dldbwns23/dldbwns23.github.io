---
layout: post
title: Deep Detection Project
date: 2026-01-07 10:00:00 +0900
tags: [Python, Project, ML/DL, CCAI]
---

from https://github.com/jeonghwan723/Deep-detection/tree/main

first trial with sample data

### dataset.py
```python
import torch
from torch.utils.data import Dataset
import numpy as np
from netCDF4 import Dataset as NC_Dataset

class ClimateDataset(Dataset):
    def __init__(self, nc_file_path, var_name_prcp='prcp', var_name_agmt='agmt', is_train=True):
        self.is_train = is_train
        self.nc_file_path = nc_file_path
        self.var_name_prcp = var_name_prcp
        self.var_name_agmt = var_name_agmt

        # Input data: Precipitation
        # Latitude range: -62.5 to 76.5 (indices 12 to 67)
        with NC_Dataset(nc_file_path['prcp'], 'r') as f:
            if var_name_prcp not in f.variables:
                possible_vars = ['pr', 'precip', 'precipitation', 'prcp']
                for name in possible_vars:
                    if name in f.variables:
                        var_name_prcp = name
                        break
            
            self.length = f.variables[var_name_prcp].shape[0]
        
        self.y_data = None    
        # Label data: AGMT (Annual Global Mean 2m air Temperature)
        if self.is_train and nc_file_path.get('agmt'):
            with NC_Dataset(nc_file_path['agmt'], 'r') as f:
                # (Time, 1) shape
                raw_y = f.variables[self.var_name_agmt][:,:,0,0].flatten()
                self.y_data = torch.from_numpy(raw_y.astype(np.float32))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        with NC_Dataset(self.nc_file_path['prcp'], 'r') as f:
            raw_data = f.variables[self.var_name_prcp][idx, :, 12:67, :]  # (Channels, Lat, Lon)

        extended_data = np.append(raw_data, raw_data[:,:,:16], axis=2)
        x = torch.from_numpy(extended_data.astype(np.float32))    

        if self.y_data is not None:
            y = self.y_data[idx].unsqueeze(0)
            return x, y

        else:
            return x
```  

### model.py
```python
import torch
import torch.nn as nn

class DDModel(nn.Module):
    def __init__(self, input_channels=1):
        super(DDModel, self).__init__()

        # Conv Block 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # Conv Block 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # Conv Block 5
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * 13 * 40, 32)
        self.act_fc1 = nn.Sigmoid()

        # Output layer for AGMT prediction
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = self.flatten(x)
        x = self.act_fc1(self.fc1(x))
        x = self.fc2(x)

        return x
    

if __name__ == "__main__":
    # Test the model with a dummy input
    dummy_input = torch.randn(1, 1, 55, 160)  # Batch size of 1, 1 channel, 160x55 spatial dimensions
    model = DDModel()

    output = model(dummy_input)
    print(output.shape)  # Expected output shape: (1, 1)
```



### main.py
```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ClimateDataset
from model import DDModel

CONFIG = {
    'exp_name': 'first_experiment',
    'n_ens': 1,
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'path': {
        'tr_prcp': 'C:/Users/eugin/Deep-Detection-Project/data/CESM2_LE_prcp_lat_tr.nc',
        'tr_agmt': 'C:/Users/eugin/Deep-Detection-Project/data/CESM2_LE_agmt_tr.nc',
        'val_prcp': 'C:/Users/eugin/Deep-Detection-Project/data/CESM2_LE_prcp_lat_val.nc',
        'val_agmt': 'C:/Users/eugin/Deep-Detection-Project/data/CESM2_LE_agmt_val.nc',
    }
}

def train():
    print(f"device: {CONFIG['device']}")
    print(f"Experiment Name: {CONFIG['exp_name']}")

    # 1. Data Loading
    try:
        train_ds = ClimateDataset(
            {'prcp': CONFIG['path']['tr_prcp'], 'agmt': CONFIG['path']['tr_agmt']}, is_train=True)
        
        val_ds = ClimateDataset(
            {'prcp': CONFIG['path']['val_prcp'], 'agmt': CONFIG['path']['val_agmt']}, is_train=True)
        
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        return
    
    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    # Model, Loss, Optimizer
    sample_x, _ = train_ds[0]
    input_channels = sample_x.shape[0]
    print(f"Input channels: {input_channels}, Input shape: {sample_x.shape}")

    model = DDModel(input_channels=input_channels).to(CONFIG['device'])
    criterion = nn.L1Loss()
    optimzer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # Training Loop
    for epoch in range(CONFIG['epochs']):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        print(f"\n[Epoch {epoch+1} / {CONFIG['epochs']}] Training Started...")
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            
            optimzer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimzer.step()
            
            train_loss += loss.item()

            if (i + 1) % 2000 == 0:
                print(f"[Train] Epoch [{epoch+1}/{CONFIG['epochs']}] | Step [{i+1}/{len(train_loader)}]")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        
        print(f"[Epoch {epoch+1} / {CONFIG['epochs']}] Validation Started...")
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()

                if (i + 1) % 2000 == 0:
                    print(f"[Val]   Epoch [{epoch+1}/{CONFIG['epochs']}] | Step [{i+1}/{len(val_loader)}]")

        # --- Epoch Result ---
        print(f"==> End of Epoch {epoch+1} | Avg Train Loss: {train_loss/len(train_loader):.4f} | Avg Val Loss: {val_loss/len(val_loader):.4f}")

    # Save Model
    os.makedirs('output', exist_ok=True)
    torch.save(model.state_dict(), 'output/model_final.pth')
    print("Training complete. Model saved to 'output/model_final.pth'")
        
if __name__ == "__main__":
    train()
```

It took more than 12 hours to run 10 epochs with batch_size=32 n_ens=1.

The result:
device: cuda
Experiment Name: first_experiment
Input channels: 1, Input shape: torch.Size([1, 55, 160])

[Epoch 1 / 10] Training Started...
[Train] Epoch [1/10] | Step [2000/22813]
[Train] Epoch [1/10] | Step [4000/22813]
[Train] Epoch [1/10] | Step [6000/22813]
[Train] Epoch [1/10] | Step [8000/22813]
[Train] Epoch [1/10] | Step [10000/22813]
[Train] Epoch [1/10] | Step [12000/22813]
[Train] Epoch [1/10] | Step [14000/22813]
[Train] Epoch [1/10] | Step [16000/22813]
[Train] Epoch [1/10] | Step [18000/22813]
[Train] Epoch [1/10] | Step [20000/22813]
[Train] Epoch [1/10] | Step [22000/22813]
[Epoch 1 / 10] Validation Started...
[Val]   Epoch [1/10] | Step [2000/4563]
[Val]   Epoch [1/10] | Step [4000/4563]
==> End of Epoch 1 | Avg Train Loss: 0.0493 | Avg Val Loss: 0.0451

[Epoch 2 / 10] Training Started...
[Train] Epoch [2/10] | Step [2000/22813]
[Train] Epoch [2/10] | Step [4000/22813]
[Train] Epoch [2/10] | Step [6000/22813]
[Train] Epoch [2/10] | Step [8000/22813]
[Train] Epoch [2/10] | Step [10000/22813]
[Train] Epoch [2/10] | Step [12000/22813]
[Train] Epoch [2/10] | Step [14000/22813]
[Train] Epoch [2/10] | Step [16000/22813]
[Train] Epoch [2/10] | Step [18000/22813]
[Train] Epoch [2/10] | Step [20000/22813]
[Train] Epoch [2/10] | Step [22000/22813]
[Epoch 2 / 10] Validation Started...
[Val]   Epoch [2/10] | Step [2000/4563]
[Val]   Epoch [2/10] | Step [4000/4563]
==> End of Epoch 2 | Avg Train Loss: 0.0460 | Avg Val Loss: 0.0449

[Epoch 3 / 10] Training Started...
[Train] Epoch [3/10] | Step [2000/22813]
[Train] Epoch [3/10] | Step [4000/22813]
[Train] Epoch [3/10] | Step [6000/22813]
[Train] Epoch [3/10] | Step [8000/22813]
[Train] Epoch [3/10] | Step [10000/22813]
[Train] Epoch [3/10] | Step [12000/22813]
[Train] Epoch [3/10] | Step [14000/22813]
[Train] Epoch [3/10] | Step [16000/22813]
[Train] Epoch [3/10] | Step [18000/22813]
[Train] Epoch [3/10] | Step [20000/22813]
[Train] Epoch [3/10] | Step [22000/22813]
[Epoch 3 / 10] Validation Started...
[Val]   Epoch [3/10] | Step [2000/4563]
[Val]   Epoch [3/10] | Step [4000/4563]
==> End of Epoch 3 | Avg Train Loss: 0.0457 | Avg Val Loss: 0.0459

[Epoch 4 / 10] Training Started...
[Train] Epoch [4/10] | Step [2000/22813]
[Train] Epoch [4/10] | Step [4000/22813]
[Train] Epoch [4/10] | Step [6000/22813]
[Train] Epoch [4/10] | Step [8000/22813]
[Train] Epoch [4/10] | Step [10000/22813]
[Train] Epoch [4/10] | Step [12000/22813]
[Train] Epoch [4/10] | Step [14000/22813]
[Train] Epoch [4/10] | Step [16000/22813]
[Train] Epoch [4/10] | Step [18000/22813]
[Train] Epoch [4/10] | Step [20000/22813]
[Train] Epoch [4/10] | Step [22000/22813]
[Epoch 4 / 10] Validation Started...
[Val]   Epoch [4/10] | Step [2000/4563]
[Val]   Epoch [4/10] | Step [4000/4563]
==> End of Epoch 4 | Avg Train Loss: 0.0453 | Avg Val Loss: 0.0457

[Epoch 5 / 10] Training Started...
[Train] Epoch [5/10] | Step [2000/22813]
[Train] Epoch [5/10] | Step [4000/22813]
[Train] Epoch [5/10] | Step [6000/22813]
[Train] Epoch [5/10] | Step [8000/22813]
[Train] Epoch [5/10] | Step [10000/22813]
[Train] Epoch [5/10] | Step [12000/22813]
[Train] Epoch [5/10] | Step [14000/22813]
[Train] Epoch [5/10] | Step [16000/22813]
[Train] Epoch [5/10] | Step [18000/22813]
[Train] Epoch [5/10] | Step [20000/22813]
[Train] Epoch [5/10] | Step [22000/22813]
[Epoch 5 / 10] Validation Started...
[Val]   Epoch [5/10] | Step [2000/4563]
[Val]   Epoch [5/10] | Step [4000/4563]
==> End of Epoch 5 | Avg Train Loss: 0.0455 | Avg Val Loss: 0.0449

[Epoch 6 / 10] Training Started...
[Train] Epoch [6/10] | Step [2000/22813]
[Train] Epoch [6/10] | Step [4000/22813]
[Train] Epoch [6/10] | Step [6000/22813]
[Train] Epoch [6/10] | Step [8000/22813]
[Train] Epoch [6/10] | Step [10000/22813]
[Train] Epoch [6/10] | Step [12000/22813]
[Train] Epoch [6/10] | Step [14000/22813]
[Train] Epoch [6/10] | Step [16000/22813]
[Train] Epoch [6/10] | Step [18000/22813]
[Train] Epoch [6/10] | Step [20000/22813]
[Train] Epoch [6/10] | Step [22000/22813]
[Epoch 6 / 10] Validation Started...
[Val]   Epoch [6/10] | Step [2000/4563]
[Val]   Epoch [6/10] | Step [4000/4563]
==> End of Epoch 6 | Avg Train Loss: 0.0453 | Avg Val Loss: 0.0473

[Epoch 7 / 10] Training Started...
[Train] Epoch [7/10] | Step [2000/22813]
[Train] Epoch [7/10] | Step [4000/22813]
[Train] Epoch [7/10] | Step [6000/22813]
[Train] Epoch [7/10] | Step [8000/22813]
[Train] Epoch [7/10] | Step [10000/22813]
[Train] Epoch [7/10] | Step [12000/22813]
[Train] Epoch [7/10] | Step [14000/22813]
[Train] Epoch [7/10] | Step [16000/22813]
[Train] Epoch [7/10] | Step [18000/22813]
[Train] Epoch [7/10] | Step [20000/22813]
[Train] Epoch [7/10] | Step [22000/22813]
[Epoch 7 / 10] Validation Started...
[Val]   Epoch [7/10] | Step [2000/4563]
[Val]   Epoch [7/10] | Step [4000/4563]
==> End of Epoch 7 | Avg Train Loss: 0.0454 | Avg Val Loss: 0.0452

[Epoch 8 / 10] Training Started...
[Train] Epoch [8/10] | Step [2000/22813]
[Train] Epoch [8/10] | Step [4000/22813]
[Train] Epoch [8/10] | Step [6000/22813]
[Train] Epoch [8/10] | Step [8000/22813]
[Train] Epoch [8/10] | Step [10000/22813]
[Train] Epoch [8/10] | Step [12000/22813]
[Train] Epoch [8/10] | Step [14000/22813]
[Train] Epoch [8/10] | Step [16000/22813]
[Train] Epoch [8/10] | Step [18000/22813]
[Train] Epoch [8/10] | Step [20000/22813]
[Train] Epoch [8/10] | Step [22000/22813]
[Epoch 8 / 10] Validation Started...
[Val]   Epoch [8/10] | Step [2000/4563]
[Val]   Epoch [8/10] | Step [4000/4563]
==> End of Epoch 8 | Avg Train Loss: 0.0452 | Avg Val Loss: 0.0448

[Epoch 9 / 10] Training Started...
[Train] Epoch [9/10] | Step [2000/22813]
[Train] Epoch [9/10] | Step [4000/22813]
[Train] Epoch [9/10] | Step [6000/22813]
[Train] Epoch [9/10] | Step [8000/22813]
[Train] Epoch [9/10] | Step [10000/22813]
[Train] Epoch [9/10] | Step [12000/22813]
[Train] Epoch [9/10] | Step [14000/22813]
[Train] Epoch [9/10] | Step [16000/22813]
[Train] Epoch [9/10] | Step [18000/22813]
[Train] Epoch [9/10] | Step [20000/22813]
[Train] Epoch [9/10] | Step [22000/22813]
[Epoch 9 / 10] Validation Started...
[Val]   Epoch [9/10] | Step [2000/4563]
[Val]   Epoch [9/10] | Step [4000/4563]
==> End of Epoch 9 | Avg Train Loss: 0.0452 | Avg Val Loss: 0.0451

[Epoch 10 / 10] Training Started...
[Train] Epoch [10/10] | Step [2000/22813]
[Train] Epoch [10/10] | Step [4000/22813]
[Train] Epoch [10/10] | Step [6000/22813]
[Train] Epoch [10/10] | Step [8000/22813]
[Train] Epoch [10/10] | Step [10000/22813]
[Train] Epoch [10/10] | Step [12000/22813]
[Train] Epoch [10/10] | Step [14000/22813]
[Train] Epoch [10/10] | Step [16000/22813]
[Train] Epoch [10/10] | Step [18000/22813]
[Train] Epoch [10/10] | Step [20000/22813]
[Train] Epoch [10/10] | Step [22000/22813]
[Epoch 10 / 10] Validation Started...
[Val]   Epoch [10/10] | Step [2000/4563]
[Val]   Epoch [10/10] | Step [4000/4563]
==> End of Epoch 10 | Avg Train Loss: 0.0455 | Avg Val Loss: 0.0457
Training complete. Model saved to 'output/model_final.pth'


No overfitting but too small epochs.
