# Menjalankan training di laptop lain

Panduan singkat untuk teman yang membantu training. Aturan lengkapnya ada di `CLAUDE.md` (terutama bagian 4 dan 10). Kalau ada langkah yang gagal atau berbeda dari panduan ini, **berhenti dan kabari pemilik**; jangan mengubah config, split, atau seed sendiri.

Untuk melanjutkan run Wav2Vec2-BERT beku yang sedang berjalan, cara termudah adalah notebook `notebooks/lanjut_training_laptop_lain.ipynb`. Notebook itu menjalankan langkah-langkah di bawah secara berurutan.

## 1. Yang perlu dibawa

| Barang | Cara | Catatan |
|---|---|---|
| Kode | `git clone https://github.com/kanzennn/wav2vec2-bert-hijaiyah.git` | lewat git supaya commit hash tercatat di run; jangan salin folder `src/` secara manual |
| `data/labels/`, `data/splits/` | sudah ada di repo | jangan diubah |
| Model `pretrained/facebook-w2v-bert-2.0/` | salin dari laptop pemilik (Google Drive folder "Training Model") | revisi `da985ba`; **jangan unduh ulang dari Hugging Face**, supaya versinya pasti sama |
| Cache `data/interim/` (`trim16k_35ecb48e72/`, `features/w2vbert_fbank_27f82ee0d4/`) | salin dari Drive | opsional, tapi menghemat waktu dan membuat data mentah tidak perlu |
| Data mentah | zip dataset → `data/raw/Data Benar/`, `data/raw/Data Salah/` | hanya bila cache `data/interim/` tidak dibawa |
| Folder run yang mau dilanjutkan | `runs/<run_id>/` | hanya untuk `--resume` |

Google Drive **hanya untuk transfer**. Salin semuanya ke SSD lokal (misalnya `D:\tugas-akhir`); **jangan jalankan training dari folder Google Drive yang sedang sync**, karena berkas besar yang sedang di-sync bisa rusak atau terkunci.

Data mentah, `pretrained/`, `data/interim/`, dan `checkpoints/` tidak boleh di-commit (sudah ada di `.gitignore`).

## 2. Siapkan environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pip install -e . --no-deps
nvidia-smi
```

- Python harus **3.11**. torch harus build CUDA 12.4 (`2.6.0+cu124`); lockfile mengambilnya dari index PyTorch.
- GPU harus NVIDIA dengan dukungan bf16 (RTX seri 30/40 atau lebih baru).
- **VRAM kurang dari 8 GB:**
  - Untuk **run baru**: boleh menurunkan `train.batch_size` dan menaikkan `train.grad_accum` supaya batch efektif tetap 32 (misalnya 4 × 8). Lakukan perubahan ini di salinan config baru, lalu catat di `reports/CHANGELOG.md` (CLAUDE.md 10.3).
  - Untuk **run yang dilanjutkan dengan `--resume`**: config tidak boleh diubah. Kabari pemilik dulu.

## 3. Urutan menjalankan

```powershell
$env:PYTHONPATH = "src"
pytest -p no:cacheprovider
python -m ta.train --config configs/w2v_bert_frozen.yaml --smoke
python -m ta.train --config configs/w2v_bert_frozen.yaml --task all
```

1. Semua tes harus lulus. Kalau ada yang gagal, berhenti dan kabari pemilik.
2. Jalankan `--smoke` dulu (subset kecil, 2 epoch) untuk memastikan semuanya jalan.
3. Setelah smoke lulus, jalankan `--task all`.

Untuk **melanjutkan run yang terhenti** (atau run yang dibawa dari laptop pemilik):

```powershell
python -m ta.train --resume runs/<run_id>
```

Selama training:
- Matikan sleep/hibernate, colokkan charger, dan tutup aplikasi lain yang memakai GPU (Discord, browser dengan akselerasi GPU, game).
- Training bisa dihentikan kapan saja (Ctrl+C); lanjutkan dengan `--resume`. Unit yang sudah selesai tidak diulang. Unit yang setengah jalan dilanjutkan dari checkpoint epoch terakhir.

## 4. Membawa hasil pulang

- Kirim `runs/<run_id>/` **tanpa** `checkpoints/` (ukurannya beberapa MB). `checkpoints/` berisi berkas sekitar 2,3 GB dan hanya berguna kalau run belum selesai.
- **Jangan menimpa `reports/results.csv` milik pemilik.** Baris hasil run ada di `runs/<run_id>/results_rows.csv`; pemilik menggabungkannya dengan:

```powershell
$env:PYTHONPATH = "src"
python -c "import pandas as pd; from ta.evaluate import append_results; append_results(pd.read_csv('runs/<run_id>/results_rows.csv').to_dict('records'))"
```

  `append_results` membuang baris lama dengan `run_id` yang sama sebelum menambahkan baris baru, jadi aman dijalankan ulang.

## 5. Yang dicatat

- Metadata mesin tercatat otomatis: hostname, OS, Python, versi library, GPU, VRAM, driver, dan versi CUDA. Lokasinya:
  - run baru: `run_args.json`;
  - tiap kali run dilanjutkan: `run_args.json` → `resumes[]`;
  - tiap unit deep learning yang selesai: `units/*.json` → `machine`;
  - run yang selesai: `metrics.json`.
- Pemilik menambahkan entri di `reports/CHANGELOG.md`: tanggal, mesin, GPU, VRAM, driver/CUDA, versi torch, run_id, dan unit mana yang dilatih di mesin mana.
- CUDA tidak 100% deterministik antar-GPU. Jadi, **sebisa mungkin semua seed Wav2Vec2-BERT dalam satu eksperimen dijalankan di mesin yang sama**. Kalau sebuah run terpaksa dipindah di tengah jalan, catat di CHANGELOG unit mana yang dilatih di mesin mana, lalu sebutkan sebagai keterbatasan.
