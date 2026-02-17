
# =============================================================
# LAB FINTECH (SINTÉTICO 2025, 1000 filas) — PREPROCESAMIENTO Y EDA
# Ejecuta:  python lab_fintech_sintetico_2025_1000.py
# =============================================================

import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_CSV = 'fintech_top_sintetico_2025_1000.csv'
DATA_DICT = 'fintech_top_sintetico_dictionary_1000.json'
OUTDIR = Path('./data_output_finanzas_sintetico_1000')
DATE_COL = 'Month'
ID_COLS = ['Company']
CAT_COLS = ['Country', 'Region', 'Segment', 'Subsegment', 'IsPublic', 'Ticker']
NUM_COLS = [
    'Users_M','NewUsers_K','TPV_USD_B','TakeRate_pct','Revenue_USD_M',
    'ARPU_USD','Churn_pct','Marketing_Spend_USD_M','CAC_USD','CAC_Total_USD_M',
    'Close_USD','Private_Valuation_USD_B'
]
PRICE_COLS = ['Close_USD']
SPLIT_DATE = '2025-09-01'

# 0) Carga
with open(DATA_DICT, 'r', encoding='utf-8') as f:
    meta = json.load(f)
print('Descripción:', meta.get('description', ''))
print('Filas declaradas:', meta.get('rows', 'N/A'))

df = pd.read_csv(DATA_CSV)
print('Shape:', df.shape)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
df = df.sort_values([DATE_COL] + ID_COLS).reset_index(drop=True)

# 1) Limpieza simple
for c in NUM_COLS:
    if c in df.columns and df[c].isna().any():
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())
for c in CAT_COLS:
    if c in df.columns and df[c].isna().any():
        df[c] = df[c].fillna('__MISSING__')

# 2) Retornos
for pc in PRICE_COLS:
    if pc in df.columns:
        df[pc + '_ret'] = (
            df.sort_values(['Company', DATE_COL])
              .groupby('Company')[pc]
              .pct_change()
        ).fillna(0.0)
        df[pc + '_logret'] = np.log1p(df[pc + '_ret']).fillna(0.0)

extra_num = [c for c in [pc+'_ret' for pc in PRICE_COLS] + [pc+'_logret' for pc in PRICE_COLS] if c in df.columns]

# 3) One-hot + split temporal + escalado
X = df.drop(columns=[DATE_COL] + ID_COLS, errors='ignore').copy()
X = pd.get_dummies(X, columns=[c for c in CAT_COLS if c in X.columns], drop_first=True)

cutoff = pd.to_datetime(SPLIT_DATE)
idx_train = df[DATE_COL] < cutoff
idx_test = df[DATE_COL] >= cutoff

X_train, X_test = X.loc[idx_train].copy(), X.loc[idx_test].copy()
num_in_X = [c for c in NUM_COLS + extra_num if c in X_train.columns]

scaler = StandardScaler()
if num_in_X:
    X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X])
    X_test[num_in_X]  = scaler.transform(X_test[num_in_X])

OUTDIR.mkdir(parents=True, exist_ok=True)
X_train.to_parquet(OUTDIR / 'fintech_train.parquet', index=False)
X_test.to_parquet(OUTDIR / 'fintech_test.parquet', index=False)

print('Exportado a:', OUTDIR)
