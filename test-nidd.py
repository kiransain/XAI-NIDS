import pandas as pd
import numpy as np

# --- CONFIGURATION ---
file_path = "5G-NIDD.csv"  # Swap to 5GAD.csv to compare
print(f"--- Starting Professional Audit for {file_path} ---")

try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: {file_path} not found.")
    exit()

# 1. ROBUST LABEL DETECTION
# Strips whitespace and handles case-insensitivity
label_candidates = [c for c in df.columns if c.strip().lower() == 'label']
if not label_candidates:
    print("!!! ERROR: No 'Label' column found. Check your CSV headers.")
    exit()
label_col = label_candidates[0]

# 2. EXPLICIT BINARY SPLITTING
# We normalize to ensure 'benign' and 'malicious' are the only targets
df[label_col] = df[label_col].astype(str).str.strip().str.lower()

benign = df[df[label_col] == 'benign'].copy()
malicious = df[df[label_col] == 'malicious'].copy()

# Catch cases where 'malicious' might be named differently (e.g., 'attack')
if len(malicious) == 0:
    malicious = df[df[label_col] != 'benign'].copy()

print(f"Total Rows: {len(df)} | Benign: {len(benign)} | Malicious: {len(malicious)}")

# 3. ADVANCED LEAKAGE AUDIT
print("\n" + "="*60)
print("TECHNICAL AUDIT REPORT")
print("="*60)

# Columns that are actually "Sub-Labels" (Ground Truth) and should be ignored
ground_truth_cols = ['attack type', 'attack tool', 'label', 'class']

for col in df.columns:
    col_lower = col.strip().lower()
    
    # Skip Ground Truth and Metadata
    if col_lower in ground_truth_cols:
        print(f"[*] Skipping ground-truth column: {col}")
        continue
    if col_lower in ['timestamp', 'flow id', 'src ip', 'dst ip', 'seq']:
        continue

    # nunique(dropna=False) ensures all-NaN columns are flagged as 1 unique value
    mal_unique_count = malicious[col].nunique(dropna=False)
    ben_unique_count = benign[col].nunique(dropna=False)
    
    # TYPE A: CONSTANT FEATURE (The Lab Artifact)
    if mal_unique_count == 1:
        val = malicious[col].iloc[0]
        print(f"[!] BIAS RISK: '{col}' is CONSTANT in Malicious (Value: {val})")
        
    # TYPE B: ZERO OVERLAP (Excluding NaNs)
    # We dropna() before set conversion to avoid the nan != nan trap
    mal_vals = set(malicious[col].dropna().unique())
    ben_vals = set(benign[col].dropna().unique())
    intersection = mal_vals.intersection(ben_vals)
    
    # Only report if there are actually values to compare
    if len(mal_vals) > 0 and len(ben_vals) > 0:
        if not intersection:
            print(f"[!!] CRITICAL LEAKAGE: '{col}' has ZERO non-NaN overlap.")
            print(f"     Malicious sample: {list(mal_vals)[:3]}")
            print(f"     Benign sample: {list(ben_vals)[:3]}")

    # TYPE C: STATISTICAL SEPARATION (Heuristic)
    if pd.api.types.is_numeric_dtype(df[col]):
        m_mean, b_mean = malicious[col].mean(), benign[col].mean()
        std = df[col].std()
        if std > 0 and abs(m_mean - b_mean) > (std * 3): # Lowered to 3 for more sensitivity
            print(f"[?] STATISTICAL ANOMALY: '{col}' mean is {abs(m_mean - b_mean)/std:.1f}x std dev apart.")

print("\n--- Audit Complete ---")