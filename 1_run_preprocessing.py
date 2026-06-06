"""
Cara menjalankan:
    python 1_run_preprocessing.py                   <- Jalankan preprocessing lokal
    python 1_run_preprocessing.py --folder indra   <- Jalankan untuk folder indra/
    python 1_run_preprocessing.py --folder furqon  <- Jalankan untuk folder furqon/
    python 1_run_preprocessing.py --folder ucup    <- Jalankan untuk folder ucup/
    python 1_run_preprocessing.py --merge  <- Gabungkan semua part hasil secara manual
"""

import os
import sys
import time
import re
import multiprocessing as mp
from functools import partial
from pathlib import Path

import nltk
import pandas as pd

from preprocessing_helper import preprocess, init_worker

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
# Ubah ke False jika ingin jalankan dengan data penuh
SAMPLE_MODE   = True
SAMPLE_SIZE   = 2000

BASE_DIR      = Path(__file__).parent
DATA_PATH     = BASE_DIR.parent / 'final_merge_dataset.csv'
PARTS_DIR     = BASE_DIR / 'parts'
RESULTS_DIR   = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FINAL  = RESULTS_DIR / 'preprocessed_data.csv'


def get_part_number(path: Path) -> int:
    """Mengambil nomor part dari nama file (misal: part_3.csv -> 3)."""
    match = re.search(r'part_(\d+)', path.name)
    return int(match.group(1)) if match else 999


def merge_preprocessed_files():
    """Menggabungkan seluruh preprocessed_part_*.csv yang ada di folder results."""
    print('\n' + '=' * 60)
    print('Penggabungan File Preprocessed')
    print('=' * 60)
    
    # Cari semua preprocessed_part_*.csv di folder results
    part_files = sorted(
        [p for p in RESULTS_DIR.glob('preprocessed_part_*.csv')],
        key=get_part_number
    )
    
    if not part_files:
        print('[ERROR] Tidak ditemukan file preprocessed_part_*.csv untuk digabungkan.')
        return
        
    print(f'Menemukan {len(part_files)} file part terproses untuk digabungkan:')
    for p in part_files:
        print(f'  - {p.name}')
        
    merged_dfs = []
    for p in part_files:
        try:
            merged_dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f'  [ERROR] Gagal membaca {p.name}: {e}')
            return
        
    final_df = pd.concat(merged_dfs, ignore_index=True)
    final_df.to_csv(OUTPUT_FINAL, index=False)
    print(f'\n[SUKSES] Berhasil menggabungkan {len(part_files)} part menjadi:')
    print(f'  -> {OUTPUT_FINAL} ({len(final_df):,} baris)')
    print('=' * 60)


def main():
    # Jika dipanggil dengan argumen --merge, langsung lakukan penggabungan
    if '--merge' in sys.argv:
        merge_preprocessed_files()
        return

    # Pastikan stopwords tersedia sebelum worker dibuat
    nltk.download('stopwords', quiet=True)

    print('=' * 60)
    print('Tahap 1: Preprocessing Bertahap')
    if SAMPLE_MODE:
        print(f'[SAMPLE MODE] Menggunakan maksimal {SAMPLE_SIZE:,} baris dari part pertama.')
    print('=' * 60)

    # 1. Pastikan folder parts/ berisi data. Jika kosong/belum ada, jalankan 0_split_dataset.py
    if not PARTS_DIR.exists() or not list(PARTS_DIR.glob('part_*.csv')):
        print('[INFO] Folder parts/ kosong atau belum dibuat. Menjalankan pemisahan data otomatis...')
        try:
            import importlib
            split_mod = importlib.import_module('0_split_dataset')
            split_mod.main()
        except ImportError:
            print('[ERROR] Skrip 0_split_dataset.py tidak ditemukan. Pastikan lokasinya benar.')
            return

    # 2. Ambil daftar file part di folder parts/
    part_files = sorted(
        [p for p in PARTS_DIR.glob('part_*.csv')],
        key=get_part_number
    )

    if not part_files:
        print('[ERROR] Tidak ada file part_*.csv di folder parts/')
        return

    # Jika SAMPLE_MODE, batasi pemrosesan hanya pada part pertama saja
    if SAMPLE_MODE:
        part_files = part_files[:1]

    print(f'Daftar part yang akan diproses secara lokal ({len(part_files)} file):')
    for p in part_files:
        print(f'  - {p.name}')
    
    n_workers = os.cpu_count()
    print(f'\nMenggunakan {n_workers} core CPU untuk multiprocessing.')

    # 3. Proses masing-masing part yang ada secara lokal
    for part_path in part_files:
        part_num = get_part_number(part_path)
        out_part_path = RESULTS_DIR / f'preprocessed_part_{part_num}.csv'

        # Checkpoint/Resume: Cek apakah output part ini sudah ada di folder results/
        if out_part_path.exists():
            print(f'\n[SKIP] Part {part_num} sudah diproses sebelumnya: {out_part_path.name}')
            continue

        print(f'\n[PROSES] Membaca {part_path.name} ...')
        df = pd.read_csv(part_path)
        
        if SAMPLE_MODE:
            df = df.head(SAMPLE_SIZE).copy()

        df['text_combined'] = df['Judul'].fillna('') + ' ' + df['Content'].fillna('')
        print(f'  Ukuran data: {len(df):,} baris')
        
        t0 = time.time()
        with mp.Pool(processes=n_workers, initializer=init_worker) as pool:
            df['text_clean'] = pool.map(
                partial(preprocess, use_stemming=True),
                df['text_combined'].tolist(),
                chunksize=200,
            )

        # Hapus baris yang teks bersihnya kosong
        df = df[df['text_clean'].str.strip() != ''].reset_index(drop=True)
        elapsed = time.time() - t0

        df.to_csv(out_part_path, index=False)
        print(f'  Selesai memproses part {part_num}: {len(df):,} baris dalam {elapsed/60:.2f} menit.')
        print(f'  Output part disimpan di: {out_part_path.name}')

    # 4. Penggabungan otomatis jika semua part yang diharapkan sudah selesai diproses secara lokal
    preprocessed_parts = list(RESULTS_DIR.glob('preprocessed_part_*.csv'))
    expected_parts_count = 1 if SAMPLE_MODE else 10
    
    if len(preprocessed_parts) >= expected_parts_count:
        print(f'\n[INFO] Menemukan {len(preprocessed_parts)} part terproses (lengkap). Menggabungkan secara otomatis...')
        merge_preprocessed_files()
    else:
        print('\n' + '=' * 60)
        print('[INFO] Preprocessing Lokal Selesai!')
        print(f'Baru ada {len(preprocessed_parts)}/10 part terproses di folder results/.')
        print('Untuk menggabungkan data dari semua laptop:')
        print('1. Salin file preprocessed_part_*.csv dari laptop lain ke folder results/ di laptop ini.')
        print('2. Jalankan perintah: python 1_run_preprocessing.py --merge')
        print('=' * 60)


if __name__ == '__main__':
    main()
