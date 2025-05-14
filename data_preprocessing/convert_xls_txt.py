# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 01:36:50 2024

@author: Chaokai
"""

import pandas as pd
import os

# Define the header for the text files
header = "0;0;0.45;0.45;0\n"

# please make sure your unit is mm and mN 
# Function to convert DataFrame to required text format
def dataframe_to_txt(df):
    txt_data = header
    for index, row in df.iterrows():
        txt_data += f"{row[0]};{row[1]}\n"
    return txt_data


# Function to read Excel files and convert to txt format
def process_excel_to_txt(file_path, output_file):
    try:
        # Attempt to read the file using xlrd
        df = pd.read_excel(file_path, header=None, engine='xlrd')
    except ImportError:
        # If xlrd is not installed, fall back to openpyxl
        df = pd.read_excel(file_path, header=None, engine='openpyxl')
    
    # Convert DataFrame to text format
    txt_data = dataframe_to_txt(df)
    
    # Save the text data to a file
    with open(output_file, 'w') as file:
        file.write(txt_data)

# Define folder name you want to store your pre-processed data
folder_name = 'data'

# Create the folder if it does not exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")
    
# Loop through samples from 1 to xx
for i in range(1, 1000):
    file_0deg = f'Demo_{i}_0deg.xls'
    file_90deg = f'Demo_{i}_90deg.xls'
    output_file_0deg = f'data\\{i}_X.txt'
    output_file_90deg = f'data\\{i}_Y.txt'
    
    if os.path.exists(file_0deg) and os.path.exists(file_90deg):
        process_excel_to_txt(file_0deg, output_file_0deg)
        process_excel_to_txt(file_90deg, output_file_90deg)
    else:
        print(f"Files for sample {i} not found.")
