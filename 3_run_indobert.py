"""
3_run_indobert.py
Tahap 3: Fine-tuning IndoBERT (Contextual Embedding).

Prasyarat:
    Jalankan 1_run_preprocessing.py terlebih dahulu.

Cara menjalankan:
    D:\\Env\\env\\Scripts\\python.exe 3_run_indobert.py

Output:
    results/indobert_results.csv
    results/cm_indobert.png
    results/indobert_learning_curve.png
    checkpoints/indobert_best.pt
"""

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
RANDOM_STATE       = 42
TEST_SIZE          = 0.20
VAL_RATIO          = 0.10

INDOBERT_MODEL     = 'indobenchmark/indobert-base-p1'
INDOBERT_MAX_LEN   = 256
INDOBERT_BATCH     = 16
INDOBERT_EPOCHS    = 100
INDOBERT_LR        = 2e-5
INDOBERT_PATIENCE  = 5

BASE_DIR     = Path(__file__).parent
RESULTS_DIR  = BASE_DIR / 'results'
CKPT_DIR     = BASE_DIR / 'checkpoints'
INPUT_PATH   = RESULTS_DIR / 'preprocessed_data.csv'
RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BG_DARK  = '#0f0f1a'
BG_PANEL = '#1a1a2e'


def clean_signatures_for_bert(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Hapus prefix editorial di awal berita (misal: "Jakarta, detikcom - ", "KOMPAS.com - ", "TEMPO.CO, Jakarta - ")
    text = re.sub(r'(?i)^\s*([a-z\s\.\,\-]+,\s*)?(detikcom|kompas\.com|tempo\.co|republika|cnbc)\s*[-–—:]\s*', '', text)
    
    # 2. Hapus brand names/variasinya di seluruh teks agar tidak jadi shortcut learning
    text = re.sub(r'(?i)\b(detikcom|kompas\.com|tempo\.co|detikfinance|detiknews|detiksport|detikhot|detik)\b', '', text)
    text = re.sub(r'(?i)\b(kompascom|kompas|tempo)\b', '', text)
    
    # 3. Bersihkan spasi ganda
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=INDOBERT_PATIENCE, ckpt_path='best.pt'):
        self.patience    = patience
        self.ckpt_path   = ckpt_path
        self.best_f1     = -1.0
        self.counter     = 0
        self.should_stop = False

    def step(self, val_f1, model):
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.counter = 0
            torch.save(model.state_dict(), self.ckpt_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ---------------------------------------------------------------------------
# Train / Eval loop
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        lbls = batch['label'].to(DEVICE)
        optimizer.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbls = batch['label'].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask)
            preds = out.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    f1  = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
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


def plot_learning_curve(history, save_path):
    ep_range = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_DARK)

    axes[0].plot(ep_range, history['train_loss'], color='#6C63FF', linewidth=2, label='Train Loss')
    axes[0].set_title('Training Loss', fontweight='bold', color='white')
    axes[0].set_xlabel('Epoch', color='white')
    axes[0].set_ylabel('Loss', color='white')
    axes[0].set_facecolor(BG_PANEL)
    axes[0].tick_params(colors='white')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ep_range, history['val_f1'],  color='#43D9AD', linewidth=2, label='Val Macro F1')
    axes[1].plot(ep_range, history['val_acc'], color='#FF6584', linewidth=2, label='Val Accuracy')
    axes[1].set_title('Validation Metrics', fontweight='bold', color='white')
    axes[1].set_xlabel('Epoch', color='white')
    axes[1].set_ylabel('Score', color='white')
    axes[1].set_facecolor(BG_PANEL)
    axes[1].tick_params(colors='white')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.suptitle('IndoBERT Learning Curve', fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print('=' * 60)
    print('Tahap 3: IndoBERT Fine-tuning (Contextual Embedding)')
    print(f'Device : {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')
        print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    print('=' * 60)

    if not INPUT_PATH.exists():
        print(f'[ERROR] File preprocessing tidak ditemukan: {INPUT_PATH}')
        print('  Jalankan dulu: 1_run_preprocessing.py')
        return

    df = pd.read_csv(INPUT_PATH)
    print(f'  Data dimuat: {len(df):,} baris')

    # IndoBERT menggunakan teks alami yang dibersihkan dari brand signatures,
    # tetapi mempertahankan tanda baca, stopwords, huruf kapital, dan bentuk kata alami (tidak di-stem)
    # agar struktur bahasa tetap utuh untuk model Transformer.
    raw_bert_input = (
        df['Judul'].fillna('') + ' [SEP] ' +
        df['Content'].fillna('').str[:500]
    )
    df['bert_input'] = raw_bert_input.apply(clean_signatures_for_bert)

    le     = LabelEncoder()
    y_bert = le.fit_transform(df['source'])
    label_names = list(le.classes_)

    # Split dengan seed yang sama agar comparable dengan baseline
    idx_all = np.arange(len(df))
    idx_temp, idx_test = train_test_split(
        idx_all, test_size=TEST_SIZE, stratify=y_bert, random_state=RANDOM_STATE
    )
    val_ratio_temp = VAL_RATIO / (1 - TEST_SIZE)
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=val_ratio_temp, stratify=y_bert[idx_temp], random_state=RANDOM_STATE
    )

    X_train = df['bert_input'].values[idx_train]
    X_val   = df['bert_input'].values[idx_val]
    X_test  = df['bert_input'].values[idx_test]
    y_train = y_bert[idx_train]
    y_val   = y_bert[idx_val]
    y_test  = y_bert[idx_test]

    print(f'  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}')

    tokenizer = AutoTokenizer.from_pretrained(INDOBERT_MODEL)

    train_ds = NewsDataset(X_train, y_train, tokenizer, INDOBERT_MAX_LEN)
    val_ds   = NewsDataset(X_val,   y_val,   tokenizer, INDOBERT_MAX_LEN)
    train_dl = DataLoader(train_ds, batch_size=INDOBERT_BATCH,     shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=INDOBERT_BATCH * 2, shuffle=False, num_workers=0)

    model     = AutoModelForSequenceClassification.from_pretrained(
        INDOBERT_MODEL, num_labels=len(label_names)
    ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=INDOBERT_LR, weight_decay=0.01)
    total_steps = len(train_dl) * INDOBERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    ckpt_path = str(CKPT_DIR / 'indobert_best.pt')
    stopper   = EarlyStopping(patience=INDOBERT_PATIENCE, ckpt_path=ckpt_path)
    history   = {'train_loss': [], 'val_f1': [], 'val_acc': []}

    t_start = time.time()
    for epoch in range(1, INDOBERT_EPOCHS + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_dl, optimizer, scheduler)
        val_f1, val_acc, _, _ = eval_epoch(model, val_dl)
        elapsed    = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)

        print(f'  Epoch {epoch:3d}/{INDOBERT_EPOCHS} | '
              f'Loss: {train_loss:.4f} | '
              f'Val F1: {val_f1:.4f} | '
              f'Val Acc: {val_acc:.4f} | '
              f'{elapsed:.1f}s | '
              f'Patience: {stopper.counter}/{stopper.patience}')

        stopper.step(val_f1, model)
        if stopper.should_stop:
            print(f'  Early stopping pada epoch {epoch}. Best Val F1: {stopper.best_f1:.4f}')
            break

    total_time = time.time() - t_start

    # Evaluasi pada test set menggunakan bobot terbaik
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    test_ds = NewsDataset(X_test, y_test, tokenizer, INDOBERT_MAX_LEN)
    test_dl = DataLoader(test_ds, batch_size=INDOBERT_BATCH * 2, shuffle=False, num_workers=0)
    test_f1, test_acc, test_preds, test_labels = eval_epoch(model, test_dl)

    print(f'\n  [IndoBERT Test] Macro F1: {test_f1:.4f} | Acc: {test_acc:.4f}')
    print(classification_report(test_labels, test_preds, target_names=label_names))
    print(f'  Total training time: {total_time / 60:.1f} menit')

    plot_cm(test_labels, test_preds, label_names,
            'IndoBERT — Test Set', RESULTS_DIR / 'cm_indobert.png')
    plot_learning_curve(history, RESULTS_DIR / 'indobert_learning_curve.png')

    result_df = pd.DataFrame([{
        'Model': 'IndoBERT', 'SMOTE': False,
        'Test_F1': round(test_f1, 4), 'Test_Acc': round(test_acc, 4),
        'CV_F1': None, 'CV_std': None,
        'Train_Time_s': round(total_time, 2),
    }])
    out_path = RESULTS_DIR / 'indobert_results.csv'
    result_df.to_csv(out_path, index=False)
    print(f'  Hasil disimpan ke: {out_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
