"""
4_generate_report.py
Tahap 4: Baca hasil dari baseline dan IndoBERT, buat grafik perbandingan final.

Prasyarat:
    Jalankan 2_run_baselines.py dan 3_run_indobert.py terlebih dahulu.

Cara menjalankan:
    D:\\Env\\env\\Scripts\\python.exe 4_generate_report.py

Output:
    results/final_results.csv
    results/final_comparison.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

BG_DARK  = '#0f0f1a'
BG_PANEL = '#1a1a2e'


def main():
    print('=' * 60)
    print('Tahap 4: Generate Laporan Akhir')
    print('=' * 60)

    baseline_path  = RESULTS_DIR / 'baseline_results.csv'
    indobert_path  = RESULTS_DIR / 'indobert_results.csv'

    missing = [p for p in [baseline_path, indobert_path] if not p.exists()]
    if missing:
        for p in missing:
            print(f'[ERROR] File tidak ditemukan: {p}')
        print('  Pastikan sudah menjalankan 2_run_baselines.py dan 3_run_indobert.py.')
        return

    # Gabungkan hasil dari kedua skrip
    baseline_df  = pd.read_csv(baseline_path)
    indobert_df  = pd.read_csv(indobert_path)
    final_df     = pd.concat([baseline_df, indobert_df], ignore_index=True).round(4)

    # Format kolom SMOTE
    final_df['SMOTE'] = final_df['SMOTE'].map({True: 'Yes', False: 'No', 'Yes': 'Yes', 'No': 'No'})
    final_df_sorted   = final_df.sort_values('Test_F1', ascending=False).reset_index(drop=True)

    out_path = RESULTS_DIR / 'final_results.csv'
    final_df_sorted.to_csv(out_path, index=False)

    print('\nRangkuman Performa Semua Model:')
    print(final_df_sorted[['Model', 'SMOTE', 'Test_F1', 'Test_Acc', 'CV_F1']].to_string(index=False))

    # Grafik perbandingan Macro F1
    labels  = [
        f"{r['Model']}\n({'SMOTE' if r['SMOTE'] == 'Yes' else 'No SMOTE'})"
        for _, r in final_df_sorted.iterrows()
    ]
    palette = plt.cm.viridis(np.linspace(0.2, 0.9, len(final_df_sorted)))

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_PANEL)
    bars = ax.bar(labels, final_df_sorted['Test_F1'], color=palette, edgecolor='none', width=0.6)
    for b, v in zip(bars, final_df_sorted['Test_F1']):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                f'{v:.4f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='white')
    ax.set_ylim(0, 1.10)
    ax.set_ylabel('Macro F1 Score (Test Set)', fontsize=12, color='white')
    ax.set_title('Perbandingan Semua Model — Test Set Macro F1',
                 fontsize=14, fontweight='bold', pad=15, color='white')
    ax.tick_params(axis='x', labelsize=9, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    chart_path = RESULTS_DIR / 'final_comparison.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()

    best = final_df_sorted.iloc[0]
    print(f'\n  Model terbaik  : {best["Model"]} (SMOTE: {best["SMOTE"]})')
    print(f'  Test Macro F1  : {best["Test_F1"]:.4f}')
    print(f'  Test Accuracy  : {best["Test_Acc"]:.4f}')
    print(f'\n  Output CSV     : {out_path}')
    print(f'  Output Chart   : {chart_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
