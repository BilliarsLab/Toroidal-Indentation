# -*- coding: utf-8 -*-
"""
Created on Wed May 14 19:12:04 2024

@author: Chaokai Zhang
"""

import numpy as np
import pandas as pd

# --- User Input ---
Sample_id = '1'

Probe_R1 = 1   # Smaller radius in µm
Probe_R2 = 11  # Larger radius in µm

### Put your testing moduli here
E_Para = 10.0  # Parallel direction Young's modulus in kPa
E_Perp = 5.0   # Perpendicular direction Young's modulus in kPa

# Set to 1 if ASP <= 10
f2 = 1

# Calculations
E_eff_0 = E_Para * 1.33
E_eff_90 = E_Perp * 1.33

Re = np.sqrt(Probe_R1 * Probe_R2)
Max_D = 0.5 * Probe_R1

D = np.linspace(0, Max_D, 101)

F0 = (4/3) * E_eff_0 * D**1.5 * np.sqrt(Re) / (f2**1.5)
F90 = (4/3) * E_eff_90 * D**1.5 * np.sqrt(Re) / (f2**1.5)

Fit_Data_0 = np.column_stack((D, F0))
Fit_Data_90 = np.column_stack((D, F90))

# Save to Excel without header or index
pd.DataFrame(Fit_Data_0).to_excel(f'Demo_{Sample_id}_0deg.xlsx', index=False, header=False)
pd.DataFrame(Fit_Data_90).to_excel(f'Demo_{Sample_id}_90deg.xlsx', index=False, header=False)
