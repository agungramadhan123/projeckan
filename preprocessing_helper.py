"""
preprocessing_helper.py
Modul bantu untuk preprocessing teks Bahasa Indonesia.
Diimpor oleh 1_run_preprocessing.py dan digunakan oleh worker multiprocessing.
"""

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------
try:
    from nltk.corpus import stopwords
    sw_id = set(stopwords.words('indonesian'))
    sw_en = set(stopwords.words('english'))
except LookupError:
    # Jika belum di-download, akan di-download oleh skrip utama sebelum worker dibuat
    sw_id = set()
    sw_en = set()

custom_sw = {
    'jakarta', 'juga', 'ini', 'itu', 'yang', 'dan', 'di', 'ke', 'dari',
    'dengan', 'untuk', 'pada', 'tidak', 'adalah', 'sudah', 'telah',
    'akan', 'ada', 'oleh', 'kami', 'mereka', 'kita', 'saya', 'anda',
    'kompascom', 'tempo', 'detikcom', 'detik', 'kompas', 'com',
    'baca', 'artikel', 'halaman', 'berikut', 'selanjutnya', 'lihat',
    'foto', 'video', 'simak', 'republika', 'cnbc', 'liputan', 'tribun',
    'media', 'online', 'news', 'id', 'http', 'https', 'www', 'co', 'go',
    'iklan', 'klik', 'scroll', 'melanjutkan', 'membaca', 'pilihan',
    'editor', 'gambas', 'lanjutkan', 'selengkapnya', 'terkini', 'berita',
    'net', 'org',
}

ALL_SW = sw_id | sw_en | custom_sw

# ---------------------------------------------------------------------------
# Worker state (diinisialisasi per-proses oleh init_worker)
# ---------------------------------------------------------------------------
stemmer    = None
stem_cache = {}


def init_worker():
    """Inisialisasi Sastrawi stemmer dan cache di setiap worker process."""
    global stemmer, stem_cache
    stemmer    = StemmerFactory().create_stemmer()
    stem_cache = {}


def preprocess(text, use_stemming=True):
    """Pipeline preprocessing lengkap untuk satu teks."""
    global stemmer, stem_cache

    if not isinstance(text, str) or text.strip() == '':
        return ''

    # Normalisasi teks
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)         # hapus URL
    text = text.encode('ascii', 'ignore').decode('ascii')  # hapus non-ASCII
    text = re.sub(r'\d+', ' ', text)                       # hapus angka
    text = re.sub(r'[^\w\s]', ' ', text)                   # hapus tanda baca
    text = text.replace('_', ' ')

    # Tokenisasi dan filter stopwords
    tokens = [t for t in text.split() if t not in ALL_SW and len(t) > 2]

    if use_stemming:
        # Inisialisasi fallback jika worker belum memanggil init_worker
        if stemmer is None:
            stemmer = StemmerFactory().create_stemmer()

        stemmed = []
        for t in tokens:
            if t not in stem_cache:
                stem_cache[t] = stemmer.stem(t)
            stemmed.append(stem_cache[t])
        tokens = stemmed

    return ' '.join(tokens)
