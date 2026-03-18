import pandas as pd
import numpy as np
import os

def audit_dataset(file_path, name):
    print(f"\n{'='*20} Auditing: {name} {'='*20}")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    # Load dataset
    df = pd.read_csv(file_path, low_memory=False)

    # Drop accidental index column if present
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    initial_count = len(df)
    print(f"Total Rows: {initial_count}")

    # 1. Check duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate Rows: {duplicates} ({(duplicates/initial_count)*100:.2f}%)")

    # 2. Identify label column safely
    possible_labels = [c for c in df.columns if c.lower() == 'label']
    if not possible_labels:
        print("ERROR: Label column not found.")
        return

    label_col = possible_labels[0]

    # Normalize labels
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    BENIGN_LABELS = {
    '5GAD': ['normal'],
    '5GC_PFCP': ['normal'],
    '5G-NIDD': ['benign']
}
    
    possible_labels = [c for c in df.columns if c.strip().lower() == 'label']
    if not possible_labels:
        print(f"ERROR: No label column found in {name}")
        return
    label_col = possible_labels[0]  # preserves original casing

    # Then safely detect benign vs malicious
    benign_mask = df[label_col].str.lower().isin([l.lower() for l in BENIGN_LABELS.get(name, ['benign'])])
    malicious_mask = ~benign_mask

    benign_df = df[benign_mask]
    malicious_df = df[malicious_mask]

    print(f"Benign Samples: {len(benign_df)}")
    print(f"Malicious Samples: {len(malicious_df)}")

    if malicious_df.empty:
        print("No malicious samples found.")
        return

    print("\nChecking for zero-variance and leakage risks:")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col == label_col:
            continue

        mal_unique = malicious_df[col].nunique(dropna=False)
        ben_unique = benign_df[col].nunique(dropna=False)

        mal_nan_ratio = malicious_df[col].isna().mean()
        ben_nan_ratio = benign_df[col].isna().mean()

        # Case 1: Constant in malicious only
        if mal_unique == 1 and ben_unique > 1:
            val = malicious_df[col].iloc[0]
            print(f"  [!] High Risk: '{col}' constant in Malicious ({val}) but variable in Benign.")

        # Case 2: Missingness leakage
        if mal_nan_ratio == 1.0 and ben_nan_ratio < 1.0:
            print(f"  [!] High Risk: '{col}' is ALL NaN in Malicious but not in Benign.")

        # Case 3: Globally constant (useless feature)
        if df[col].nunique(dropna=False) == 1:
            print(f"  [-] Useless Feature: '{col}' constant across entire dataset.")

    print("Audit complete.")

audit_dataset('5G-NIDD.csv', '5G-NIDD')
audit_dataset('5GAD.csv', '5GAD')
audit_dataset('5GC_PFCP.csv', '5GC_PFCP')

