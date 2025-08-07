import os
import numpy as np
import pandas as pd
from kagglehub import KaggleDatasetAdapter
import kagglehub

def download_dataset():
    """Download dataset from Kaggle and save to data_raw folder"""
    # Dataset ID Kaggle
    dataset_id = "altruistdelhite04/loan-prediction-problem-dataset"

    # Folder penyimpanan
    os.makedirs("../data_raw", exist_ok=True)

    # Daftar file yang ingin diunduh
    file_list = ["train_u6lujuX_CVtuZ9i.csv"]

    # Loop untuk load dan simpan masing-masing file
    for file_name in file_list:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            dataset_id,
            file_name
        )
        
        # Simpan ke folder data_raw
        save_path = os.path.join("../data_raw", file_name)
        df.to_csv(save_path, index=False)
        print(f" {file_name} disimpan di: {save_path}")
    
    return df

def clean_data(df):
    """Clean and preprocess the raw data"""
    df_clean = df.copy()

    # Hapus kolom Loan_ID
    if 'Loan_ID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Loan_ID'])

    # Isi nilai null
    for col in df_clean.columns:
        if df_clean[col].dtype in [np.float64, np.int64, float, int]:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
        else:
            mode_val = df_clean[col].mode(dropna=True)
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val[0])

    # Bersihkan dan konversi kolom 'Dependents'
    if 'Dependents' in df_clean.columns:
        df_clean['Dependents'] = df_clean['Dependents'].replace('3+', 4)
        df_clean['Dependents'] = df_clean['Dependents'].astype(float)

    # Ubah 'Loan_Status' menjadi 0.0 dan 1.0
    if 'Loan_Status' in df_clean.columns:
        df_clean['Loan_Status'] = df_clean['Loan_Status'].replace({'Y': 1.0, 'N': 0.0}).astype(float)

    # Ubah 'ApplicantIncome' ke float
    if 'ApplicantIncome' in df_clean.columns:
        df_clean['ApplicantIncome'] = df_clean['ApplicantIncome'].astype(float)

    # Hapus duplikat
    df_clean = df_clean.drop_duplicates()

    return df_clean

def save_clean_data(df_clean):
    """Save cleaned data to data_clean folder"""
    os.makedirs('data_clean', exist_ok=True)
    save_path = os.path.join('data_clean', 'data_clean.csv')
    df_clean.to_csv(save_path, index=False)
    print(f"Data bersih disimpan di: {save_path}")

def main():
    """Main function to automate the data preprocessing"""
    print("Memulai proses preprocessing data...")
    
    # Step 1: Download data
    print("\n1. Mengunduh data dari Kaggle...")
    raw_data = download_dataset()
    
    # Step 2: Clean data
    print("\n2. Membersihkan data...")
    cleaned_data = clean_data(raw_data)
    
    # Step 3: Save cleaned data
    print("\n3. Menyimpan data yang sudah dibersihkan...")
    save_clean_data(cleaned_data)
    
    print("\nProses preprocessing selesai!")

if __name__ == "__main__":
    main()