"""
0_split_dataset.py
Membagi dataset final_merge_dataset.csv menjadi 10 part secara terstratifikasi (Stratified Shuffled Split)
berdasarkan kolom 'source'.

Cara menjalankan:
    python 0_split_dataset.py
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
NUM_PARTS    = 10

BASE_DIR     = Path(__file__).parent
DATA_PATH    = BASE_DIR.parent / 'final_merge_dataset.csv'
PARTS_DIR    = BASE_DIR / 'parts'


def main():
    print('=' * 60)
    print('Tahap 0: Dataset Splitting')
    print('=' * 60)

    if not DATA_PATH.exists():
        print(f'[ERROR] File dataset utama tidak ditemukan di: {DATA_PATH}')
        print('  Pastikan file final_merge_dataset.csv berada di folder induk (tubes/).')
        return

    # Buat direktori parts/ jika belum ada
    PARTS_DIR.mkdir(exist_ok=True)

    print(f'Membaca data dari: {DATA_PATH} ...')
    df = pd.read_csv(DATA_PATH)
    print(f'  Dataset dimuat: {len(df):,} baris')

    # Tangani NaN pada kolom source agar tidak error saat splitting
    df['source'] = df['source'].fillna('unknown')

    # Stratified K-Fold untuk membagi dataset menjadi N part terpisah secara seimbang
    skf = StratifiedKFold(n_splits=NUM_PARTS, shuffle=True, random_state=RANDOM_STATE)

    print(f'Membagi data menjadi {NUM_PARTS} bagian secara terstratifikasi (Stratified K-Fold)...')
    for part_idx, (_, test_idx) in enumerate(skf.split(df, df['source']), 1):
        part_df = df.iloc[test_idx].copy()
        output_path = PARTS_DIR / f'part_{part_idx}.csv'
        
        part_df.to_csv(output_path, index=False)
        
        # Cetak info distribusi label di part ini
        dist = part_df['source'].value_counts().to_dict()
        print(f'  [SAVE] {output_path.name:<12} | Baris: {len(part_df):,} | Distribusi: {dist}')

    print('=' * 60)
    print(f'Selesai! Seluruh part disimpan di folder: {PARTS_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
