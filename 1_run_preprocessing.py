"""
1_run_preprocessing.py
Tahap 1: Bersihkan teks mentah dan simpan hasilnya ke disk.

Cara menjalankan:
    D:\\Env\\env\\Scripts\\python.exe 1_run_preprocessing.py

Output:
    results/preprocessed_data.csv
"""

import os
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path

import nltk
import pandas as pd

from preprocessing_helper import preprocess, init_worker

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
# Ubah ke False jika ingin jalankan dengan data penuh (80k baris)
SAMPLE_MODE   = True
SAMPLE_SIZE   = 2_000

BASE_DIR      = Path(__file__).parent
DATA_PATH     = BASE_DIR / 'final_merge_dataset.csv'
RESULTS_DIR   = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH   = RESULTS_DIR / 'preprocessed_data.csv'


def main():
    # Pastikan stopwords tersedia sebelum worker dibuat
    nltk.download('stopwords', quiet=True)

    print('=' * 60)
    print('Tahap 1: Preprocessing')
    if SAMPLE_MODE:
        print(f'[SAMPLE MODE] Menggunakan {SAMPLE_SIZE:,} baris pertama.')
    print('=' * 60)

    # Cek cache
    if OUTPUT_PATH.exists():
        print(f'[SKIP] Cache ditemukan: {OUTPUT_PATH}')
        print('  Hapus file tersebut jika ingin preprocessing ulang.')
        return

    df = pd.read_csv(DATA_PATH)
    if SAMPLE_MODE:
        df = df.head(SAMPLE_SIZE).copy()

    df['text_combined'] = df['Judul'].fillna('') + ' ' + df['Content'].fillna('')

    n_workers = os.cpu_count()
    print(f'  Dataset: {len(df):,} baris')
    print(f'  Workers: {n_workers} threads')

    t0 = time.time()
    with mp.Pool(processes=n_workers, initializer=init_worker) as pool:
        df['text_clean'] = pool.map(
            partial(preprocess, use_stemming=True),
            df['text_combined'].tolist(),
            chunksize=200,
        )

    df = df[df['text_clean'].str.strip() != ''].reset_index(drop=True)
    elapsed = time.time() - t0

    df.to_csv(OUTPUT_PATH, index=False)

    print(f'  Selesai: {len(df):,} baris dalam {elapsed/60:.1f} menit.')
    print(f'  Output  : {OUTPUT_PATH}')
    print('=' * 60)


if __name__ == '__main__':
    main()
