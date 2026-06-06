"""
2_run_baselines.py
Tahap 2: TF-IDF + SVD + SMOTE + 4 model baseline.

Prasyarat:
    Jalankan 1_run_preprocessing.py terlebih dahulu.

Cara menjalankan:
    D:\\Env\\env\\Scripts\\python.exe 2_run_baselines.py

Output:
    results/baseline_results.csv
    results/cm_*.png
    results/split_distribution.png
    results/smote_distribution.png
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
RANDOM_STATE       = 42
TEST_SIZE          = 0.20
VAL_RATIO          = 0.10

TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE  = (1, 3)
SVD_COMPONENTS     = 300

BASE_DIR     = Path(__file__).parent
RESULTS_DIR  = BASE_DIR / 'results'
INPUT_PATH   = RESULTS_DIR / 'preprocessed_data.csv'
RESULTS_DIR.mkdir(exist_ok=True)

# Warna visualisasi
LABEL_COLORS = {'detik': '#43D9AD', 'kompas': '#6C63FF', 'tempo': '#FF6584'}
BG_DARK      = '#0f0f1a'
BG_PANEL     = '#1a1a2e'


# ---------------------------------------------------------------------------
# Helper: Split
# ---------------------------------------------------------------------------
def run_split(df):
    print('[2/4] Split data...')
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['source'])
    label_names  = list(le.classes_)
    colors       = [LABEL_COLORS[l] for l in label_names]

    X = df['text_clean'].values
    y = df['label'].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    val_ratio_temp = VAL_RATIO / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_temp, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f'  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}')

    # Visualisasi distribusi split
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(BG_DARK)
    for ax, (split_y, title) in zip(axes, [
        (y_train, f'Train ({len(y_train):,})'),
        (y_val,   f'Val ({len(y_val):,})'),
        (y_test,  f'Test ({len(y_test):,})'),
    ]):
        counts = [np.sum(split_y == i) for i in range(len(label_names))]
        ax.bar(label_names, counts, color=colors, edgecolor='none')
        for i, c in enumerate(counts):
            ax.text(i, c + 1, f'{c:,}', ha='center', fontsize=9, fontweight='bold', color='white')
        ax.set_title(title, fontweight='bold', color='white')
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors='white')
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Distribusi Label per Split', fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'split_distribution.png', dpi=150, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()

    return X_train, X_val, X_test, y_train, y_val, y_test, label_names


# ---------------------------------------------------------------------------
# Helper: TF-IDF
# ---------------------------------------------------------------------------
def run_tfidf(X_train, X_val, X_test):
    print('[2/4] TF-IDF vectorization...')
    tfidf = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        min_df=2,
        strip_accents='unicode',
        analyzer='word',
    )
    X_tr  = tfidf.fit_transform(X_train)
    X_va  = tfidf.transform(X_val)
    X_te  = tfidf.transform(X_test)
    print(f'  Vocab: {len(tfidf.vocabulary_):,} | Shape train: {X_tr.shape}')
    return tfidf, X_tr, X_va, X_te


# ---------------------------------------------------------------------------
# Helper: TruncatedSVD + SMOTE
# ---------------------------------------------------------------------------
def run_smote(X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train):
    print('[2/4] TruncatedSVD + SMOTE...')
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    X_tr_svd = svd.fit_transform(X_train_tfidf)
    X_va_svd = svd.transform(X_val_tfidf)
    X_te_svd = svd.transform(X_test_tfidf)
    print(f'  SVD explained variance: {svd.explained_variance_ratio_.sum():.4f}')

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_svd, y_train)
    print(f'  SMOTE: {len(y_train):,} -> {len(y_tr_sm):,} sampel')

    # Visualisasi SMOTE
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(BG_DARK)
    clrs = list(LABEL_COLORS.values())
    for ax, (y_arr, title) in zip(axes, [
        (y_train,  'Sebelum SMOTE'),
        (y_tr_sm,  'Sesudah SMOTE'),
    ]):
        vals, counts = np.unique(y_arr, return_counts=True)
        ax.bar(vals, counts, color=clrs[:len(vals)], edgecolor='none')
        ax.set_title(title, fontweight='bold', color='white')
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors='white')
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Distribusi Label SMOTE', fontsize=13, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'smote_distribution.png', dpi=150, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()

    return X_tr_svd, X_va_svd, X_te_svd, X_tr_sm, y_tr_sm


# ---------------------------------------------------------------------------
# Helper: Evaluasi & Confusion Matrix
# ---------------------------------------------------------------------------
def evaluate(model, X_test, y_test, label_names, tag):
    preds = model.predict(X_test)
    f1    = f1_score(y_test, preds, average='macro')
    acc   = accuracy_score(y_test, preds)
    print(f'  [{tag}] Macro F1: {f1:.4f} | Acc: {acc:.4f}')
    print(classification_report(y_test, preds, target_names=label_names))
    return preds, f1, acc


def plot_cm(y_true, y_pred, label_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_PANEL)
    im = ax.imshow(cm, cmap='plasma')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha='right', color='white')
    ax.set_yticklabels(label_names, color='white')
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] < cm.max() / 2 else 'black')
    ax.set_xlabel('Predicted', color='white')
    ax.set_ylabel('Actual', color='white')
    ax.set_title(title, fontweight='bold', pad=12, color='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()


def run_cv(estimator, X, y, model_name, cv_folds=10):
    skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        estimator, X, y, cv=skf,
        scoring={'f1_macro': 'f1_macro', 'accuracy': 'accuracy'},
        n_jobs=-1,
    )
    mean_f1  = scores['test_f1_macro'].mean()
    std_f1   = scores['test_f1_macro'].std()
    mean_acc = scores['test_accuracy'].mean()
    print(f'  [{model_name}] {cv_folds}-Fold CV — F1: {mean_f1:.4f} ± {std_f1:.4f} | Acc: {mean_acc:.4f}')
    return mean_f1, std_f1, mean_acc


# ---------------------------------------------------------------------------
# Tahap Utama: 4 Baseline Models
# ---------------------------------------------------------------------------
def run_baselines(X_train_tfidf, X_test_tfidf, y_train, y_test,
                  X_train_svd, X_test_svd, X_train_sm, y_train_sm,
                  label_names):
    print('\n[3/4] Melatih 4 model baseline...')
    results  = []
    n_jobs   = os.cpu_count()

    # --- 1. Naive Bayes ---
    print('\n  [NB] Naive Bayes (ComplementNB) — No SMOTE')
    t0  = time.time()
    nb  = ComplementNB(alpha=0.1)
    nb.fit(X_train_tfidf, y_train)
    tt  = time.time() - t0
    preds, f1, acc = evaluate(nb, X_test_tfidf, y_test, label_names, 'NB no-SMOTE')
    plot_cm(y_test, preds, label_names, 'Naive Bayes — No SMOTE', RESULTS_DIR / 'cm_nb_nosmote.png')
    cv_f1, cv_std, cv_acc = run_cv(ComplementNB(alpha=0.1), X_train_tfidf, y_train, 'NB CV')
    results.append({'Model': 'Naive Bayes', 'SMOTE': False,
                    'Test_F1': f1, 'Test_Acc': acc,
                    'CV_F1': cv_f1, 'CV_std': cv_std, 'Train_Time_s': round(tt, 2)})

    print('\n  [NB] Naive Bayes (MultinomialNB) — SMOTE')
    scaler   = MinMaxScaler()
    X_sm_sc  = scaler.fit_transform(X_train_sm)
    X_te_sc  = scaler.transform(X_test_svd)
    t0       = time.time()
    nb_sm    = MultinomialNB(alpha=0.1)
    nb_sm.fit(X_sm_sc, y_train_sm)
    tt       = time.time() - t0
    preds, f1, acc = evaluate(nb_sm, X_te_sc, y_test, label_names, 'NB SMOTE')
    plot_cm(y_test, preds, label_names, 'Naive Bayes — SMOTE', RESULTS_DIR / 'cm_nb_smote.png')
    results.append({'Model': 'Naive Bayes', 'SMOTE': True,
                    'Test_F1': f1, 'Test_Acc': acc,
                    'CV_F1': None, 'CV_std': None, 'Train_Time_s': round(tt, 2)})

    # --- 2. SVM ---
    print('\n  [SVM] LinearSVC — No SMOTE')
    t0  = time.time()
    svm = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)
    svm.fit(X_train_tfidf, y_train)
    tt  = time.time() - t0
    preds, f1, acc = evaluate(svm, X_test_tfidf, y_test, label_names, 'SVM no-SMOTE')
    plot_cm(y_test, preds, label_names, 'SVM Linear — No SMOTE', RESULTS_DIR / 'cm_svm_nosmote.png')
    cv_f1, cv_std, cv_acc = run_cv(
        LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE),
        X_train_tfidf, y_train, 'SVM CV')
    results.append({'Model': 'SVM Linear', 'SMOTE': False,
                    'Test_F1': f1, 'Test_Acc': acc,
                    'CV_F1': cv_f1, 'CV_std': cv_std, 'Train_Time_s': round(tt, 2)})

    print('\n  [SVM] LinearSVC — SMOTE')
    t0     = time.time()
    svm_sm = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    svm_sm.fit(X_train_sm, y_train_sm)
    tt     = time.time() - t0
    preds, f1, acc = evaluate(svm_sm, X_test_svd, y_test, label_names, 'SVM SMOTE')
    plot_cm(y_test, preds, label_names, 'SVM Linear — SMOTE', RESULTS_DIR / 'cm_svm_smote.png')
    results.append({'Model': 'SVM Linear', 'SMOTE': True,
                    'Test_F1': f1, 'Test_Acc': acc,
                    'CV_F1': None, 'CV_std': None, 'Train_Time_s': round(tt, 2)})

    # --- 3. Decision Tree ---
    print('\n  [DT] Decision Tree — No SMOTE')
    t0 = time.time()
    dt = DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE, min_samples_leaf=5)
    dt.fit(X_train_tfidf, y_train)
    tt = time.time() - t0
    preds, f1, acc = evaluate(dt, X_test_tfidf, y_test, label_names, 'DT no-SMOTE')
    plot_cm(y_test, preds, label_names, 'Decision Tree — No SMOTE', RESULTS_DIR / 'cm_dt_nosmote.png')
    cv_f1, cv_std, cv_acc = run_cv(
        DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE, min_samples_leaf=5),
        X_train_tfidf, y_train, 'DT CV')
    results.append({'Model': 'Decision Tree', 'SMOTE': False,
                    'Test_F1': f1, 'Test_Acc': acc,
                    'CV_F1': cv_f1, 'CV_std': cv_std, 'Train_Time_s': round(tt, 2)})

    print('\n  [DT] Decision Tree — SMOTE')
    t0    = time.time()
    dt_sm = DecisionTreeClassifier(random_state=RANDOM_STATE, min_samples_leaf=5)
    dt_sm.fit(X_train_sm, y_train_sm)
    tt    = time.time() - t0
    preds, f1, acc = evaluate(dt_sm, X_test_svd, y_test, label_names, 'DT SMOTE')
    plot_cm(y_test, preds, label_names, 'Decision Tree — SMOTE', RESULTS_DIR / 'cm_dt_smote.png')
    results.append({'Model': 'Decision Tree', 'SMOTE': True,
                    'Test_F1': f1, 'Test_Acc': acc,
                    'CV_F1': None, 'CV_std': None, 'Train_Time_s': round(tt, 2)})

    # --- 4. Random Forest ---
    print('\n  [RF] Random Forest — No SMOTE')
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=RANDOM_STATE, n_jobs=n_jobs)
    rf.fit(X_train_tfidf, y_train)
    tt = time.time() - t0
    preds, f1, acc = evaluate(rf, X_test_tfidf, y_test, label_names, 'RF no-SMOTE')
    plot_cm(y_test, preds, label_names, 'Random Forest — No SMOTE', RESULTS_DIR / 'cm_rf_nosmote.png')
    cv_f1, cv_std, cv_acc = run_cv(
        RandomForestClassifier(n_estimators=200, class_weight='balanced',
                               random_state=RANDOM_STATE, n_jobs=n_jobs),
        X_train_tfidf, y_train, 'RF CV')
    results.append({'Model': 'Random Forest', 'SMOTE': False,
                    'Test_F1': f1, 'Test_Acc': acc,
                    'CV_F1': cv_f1, 'CV_std': cv_std, 'Train_Time_s': round(tt, 2)})

    print('\n  [RF] Random Forest — SMOTE')
    t0    = time.time()
    rf_sm = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=n_jobs)
    rf_sm.fit(X_train_sm, y_train_sm)
    tt    = time.time() - t0
    preds, f1, acc = evaluate(rf_sm, X_test_svd, y_test, label_names, 'RF SMOTE')
    plot_cm(y_test, preds, label_names, 'Random Forest — SMOTE', RESULTS_DIR / 'cm_rf_smote.png')
    results.append({'Model': 'Random Forest', 'SMOTE': True,
                    'Test_F1': f1, 'Test_Acc': acc,
                    'CV_F1': None, 'CV_std': None, 'Train_Time_s': round(tt, 2)})

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print('=' * 60)
    print('Tahap 2: TF-IDF + SVD + SMOTE + Baseline Models')
    print('=' * 60)

    if not INPUT_PATH.exists():
        print(f'[ERROR] File preprocessing tidak ditemukan: {INPUT_PATH}')
        print('  Jalankan dulu: 1_run_preprocessing.py')
        return

    print(f'[1/4] Membaca data bersih dari: {INPUT_PATH}')
    df = pd.read_csv(INPUT_PATH)
    print(f'  Total baris: {len(df):,}')

    X_train, X_val, X_test, y_train, y_val, y_test, label_names = run_split(df)
    tfidf, X_train_tfidf, X_val_tfidf, X_test_tfidf = run_tfidf(X_train, X_val, X_test)
    X_train_svd, X_val_svd, X_test_svd, X_train_sm, y_train_sm = run_smote(
        X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train
    )

    results = run_baselines(
        X_train_tfidf, X_test_tfidf, y_train, y_test,
        X_train_svd, X_test_svd, X_train_sm, y_train_sm,
        label_names
    )

    out_df = pd.DataFrame(results).round(4)
    out_path = RESULTS_DIR / 'baseline_results.csv'
    out_df.to_csv(out_path, index=False)

    print(f'\n[4/4] Baseline selesai. Hasil disimpan ke: {out_path}')
    print(out_df.to_string(index=False))
    print('=' * 60)


if __name__ == '__main__':
    main()
