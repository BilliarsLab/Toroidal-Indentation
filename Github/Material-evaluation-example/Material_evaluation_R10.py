# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 09 11:05:48 2024
PCA + NN to predict anisotropic properties
@author: Chaokai Zhang
"""
#%% modulus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import os
import os.path
from scipy.interpolate import interp1d
#%%
len_of_fp = 60   # Number of feature vectors(Knots) 125
batch_s = 128 # 32
nof = 1900  # Number of pair files in the input folder
# 719
#%% Load raw data
def preprocess_data(path, nof, len_of_fp, max_depth, min_radius):
    Xs = []
    Ys = []
    for index in range(1, nof):
        xfile = os.path.join(path, f"{index}_Y.txt")
        yfile = os.path.join(path, f"{index}_X.txt")
        if (os.path.exists(xfile)) and (os.path.exists(yfile)):
            with open(xfile, 'r') as f:
                lines = f.readlines()
                if not(os.stat(xfile).st_size == 0):
                    line = lines.pop(0)
                    [Ex, Ey, nuxy, nuyz, Gxy] = list(map(float, line.split(";")))
                    xs = []; xfs = []
                    for line in lines:
                        [x, xf] = list(map(float, line.split(";")))
                        xs.append(x/1000)  # deflection
                        xfs.append(xf/1000)  # force

                new_xs = np.linspace(np.min(xs), np.max(xs), len_of_fp)
                # new_xs = np.linspace(np.min(xs), 0.0005, len_of_fp)
                new_xfs = np.interp(x=new_xs, xp=xs, fp=xfs, left=None, right=None, period=None)

                with open(yfile, 'r') as f:
                    lines = f.readlines()
                    line = lines.pop(0)
                    [Ex, Ey, nuxy, nuyz, Gxy] = list(map(float, line.split(";")))
                    ys = []
                    yfs = []
                    for line in lines:
                        [y, yf] = list(map(float, line.split(";")))
                        ys.append(y/1000)  # deflection
                        yfs.append(yf/1000)  # force

                new_ys = np.linspace(np.min(ys), np.max(ys), len_of_fp)
                # new_ys = np.linspace(np.min(ys), 0.0005, len_of_fp)
                new_yfs = np.interp(x=new_ys, xp=ys, fp=yfs, left=None, right=None, period=None)

                max_def = max(max(new_ys), max(new_xs))  # Should be equal to max_depth

                new_ys_ = new_ys / max_def  # Normalized
                new_xs_ = new_xs / max_def

                new_yfs_ = new_yfs / ((max_def ** 1.5) * (min_radius ** 0.5))
                new_xfs_ = new_xfs / ((max_def ** 1.5) * (min_radius ** 0.5))

                feature =  list(new_xfs_) +list(new_yfs_) 

                Xs.append(feature)
                Ys.append([Ex, Ey])
    # return np.array(Xs), np.array(Ys)
    return np.array(Xs), np.array(Ys)

# Constants Input

max_depth = 2 / 1000.0  # mm
min_radius = 2 / 1000.0  # mm

# Preprocess training data
train_path = "F:\\Anisotropic_Indentation_ML\\New_results\\R10"
X_train, y_train = preprocess_data(train_path, nof, len_of_fp, max_depth, min_radius)

min_radius = 0.5 / 1000.0  # mm
exp_path = "F:\\Anisotropic_Indentation_ML\\New_results\\R10_VAL"
X_val, y_val = preprocess_data(exp_path, nof, len_of_fp, max_depth, min_radius)

# Scaling the data
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_val_scaled = sc_X.transform(X_val)

n_components = 140
#% PCA and dataloader for NN
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Apply PCA with the best number of components (35)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)

#%% Define the neural network model with the best parameters
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(len_of_fp*2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.2149)  # Best dropout rate
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

model = NeuralNet().to(device)

#%% 
model = NeuralNet().to(device)
model.load_state_dict(torch.load('F:\\Anisotropic_Indentation_ML\\New_results\\R10_best_model.pth'))
model.eval()

with torch.no_grad():
    y_pred_val_tensor = model(X_val_tensor)
    y_pred_val = y_pred_val_tensor.cpu().numpy()  # Move predictions back to CPU for evaluation
    
column_titles = ['E11', 'E22']
df = pd.DataFrame(y_pred_val, columns=column_titles)
df.insert(0, 'Sample_No', range(1, 1 + len(df)))
# Adding a new column for the ratio E11/E22
df['E11/E22'] = df['E11'] / df['E22']
df.to_excel('R10_evaluation_E11_E22.xlsx', index=False)
#%% 



