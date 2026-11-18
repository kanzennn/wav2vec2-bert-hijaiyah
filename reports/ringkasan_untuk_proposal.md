# Ringkasan proses TA untuk penyusunan proposal

Dokumen ini untuk Claude chat sebagai konteks mencicil proposal/laporan TA.
Status per **16 November 2026, siang**. Semua angka diambil dari `reports/` dan `runs/` di repo; angka Wav2Vec2-BERT masih **sementara** (training belum selesai).

## Aturan untuk Claude chat

1. **Jangan mengarang angka.** Pakai hanya angka di dokumen ini. Bagian yang belum ada hasilnya (T2 Wav2Vec2-BERT, ablasi augmentasi) tulis sebagai `[menunggu hasil]`.
2. **Jangan membandingkan langsung dengan angka Azis & Lestari (2023).** Mereka memakai 12 label, akurasi, dan split 15+15 / 5+5. "Peningkatan" di TA ini hanya terhadap XGBoost yang dilatih ulang pada split yang sama.
3. Sitasi yang disebut di sini wajib dicek ulang ke sumber aslinya (judul, tahun, venue) sebelum masuk daftar pustaka.
4. Bahasa: Indonesia baku untuk proposal. Istilah teknis boleh dalam bahasa Inggris dan dicetak miring.

---

## 1. Identitas dan tujuan

- **Judul:** Peningkatan Klasifikasi Sifat Huruf Hijaiyah Menggunakan Wav2Vec2-BERT.
- **Penulis:** Faiz Elfahad Kurniawan, S1 Informatika, Telkom University.
- **Ide inti:** mengukur seberapa besar Wav2Vec2-BERT yang di-*fine-tune* meningkatkan klasifikasi **sifat huruf** (bukan identitas huruf) dari rekaman pelafalan pendek, dibanding baseline **XGBoost + MFCC**.
- **Hanya dua model.** Cakupan sengaja dikecilkan atas arahan dosen pembimbing. Model lain (SVM, RF, CNN, CRNN, wav2vec2-base) pernah dicoba lalu diarsipkan, dan tidak dilaporkan sebagai hasil.
- **Pertanyaan penelitian:**
  1. Seberapa besar peningkatan macro-F1 dan akurasi Wav2Vec2-BERT terhadap XGBoost pada tiap sifat, untuk responden yang belum pernah dilihat?
  2. Pada sifat mana peningkatannya paling besar dan paling kecil?
  3. Sifat, item, dan harakat mana yang paling sulit?
- **Definisi peningkatan:** macro-F1 Wav2Vec2-BERT − macro-F1 XGBoost, pada data uji dan fold yang sama, per sifat.
- **Di luar lingkup:** perangkat keras (ESP32, mikrofon), identifikasi huruf 36 kelas, dan pembuatan aplikasi.

## 2. Dataset

- **Sumber:** *Hijaiyah Letter Pronunciation Audio Dataset*, A. M. H. Azis, Mendeley Data V1, doi 10.17632/r5fy2dd98g.1, lisensi CC BY 4.0.
- **Isi:** Data Benar (pelafalan benar) dan Data Salah (pelafalan salah). Masing-masing 20 folder responden, dengan 540 berkas per responden (36 item × 3 harakat × 5 pengulangan).
  - Harakat: fathah, kasrah, dhammah.
  - Format audio: mono WAV 44,1 kHz (3 berkas 48 kHz).
- **36 item:** 27 huruf biasa, 3 varian huruf 25 (`25i`, `25ie`, `25ik`), dan 6 varian berakhiran `t`. Arti akhiran varian belum dijelaskan dataset dan masih ditanyakan ke penulis dataset.
- **Jumlah rekaman:**
  - Data Benar 10.791, Data Salah 10.800, total di metadata 21.591.
  - 9 berkas bernama tidak standar sudah dikeluarkan oleh dataset.
  - Kami mengeluarkan **5 rekaman** lagi karena isinya identik lintas data latih–uji.
  - **Rekaman terpakai: 21.586** (latih 17.275, uji 4.311).
- **Durasi:** median sekitar 0,83 s; persentil 1–99 = 0,23–1,82 s; minimum 0,03 s; maksimum 11,1 s.
  - 754 rekaman di bawah 0,3 s dan 110 di atas 2 s, hampir semuanya dari Data Salah.
  - Semua tetap dipakai (tidak ada yang dibuang) dan dicatat di `reports/outlier_durasi.csv`.
- **Label:** kolom `y_S1`, `y_S3`, `y_T2` dari `metadata_rekaman.csv`.
  - Label Data Salah adalah kebalikan dari label Data Benar pada item yang sama; pembalikan ini sudah dilakukan oleh dataset. Kontras benar–salah inilah yang membuat sifat dapat diklasifikasi.
  - S3 juga bergantung harakat untuk huruf خ, ر, غ: *tafkhim* kecuali dengan kasrah.
  - Label sudah diuji konsisten dengan `label_per_huruf.csv` lewat unit test.
- **Keseimbangan kelas:** porsi kelas 1 sekitar 0,50 di data latih dan uji untuk ketiga sifat, jadi *class weight* tidak dipakai.

## 3. Target klasifikasi (tiga tugas biner, satu model per sifat)

| ID | Sifat | Kelas 0 | Kelas 1 | Huruf kelas 1 (Data Benar) |
|---|---|---|---|---|
| S1 | jahr vs hams | jahr | hams | 10 huruf |
| S3 | istifal vs isti'la | istifal | isti'la | 7 huruf (خ ص ض ط ظ غ ق) |
| T2 | qalqalah | tidak ada | ada | 5 huruf (ب ج د ط ق) |

Ketiga sifat ini sama dengan yang dimodelkan penulis dataset. Kolom `is_correct` (benar/salah) disimpan untuk analisis, tetapi **tidak** dipakai sebagai fitur.

## 4. Pembagian data dan pencegahan kebocoran

- **Split bawaan dataset dipakai apa adanya** (speaker-exclusive):
  - **Uji (fold 0):** 4.311 rekaman dari responden benar-2, 9, 12, 13 dan salah-2, 4, 10, 11.
  - **Latih:** 17.275 rekaman di fold 1–5 (3.240 / 3.780 / 3.235 / 3.780 / 3.240) menurut `grup_cv`.
- **Grup responden:** 40 responden menjadi 37 grup, karena beberapa folder berisi rekaman identik dan digabung (benar-8_18_19, salah-13_15). Satu grup tidak pernah terpecah antar fold.
- **Verifikasi:**
  - Irisan responden latih–uji kosong.
  - Hash md5 isi audio yang sama di latih dan uji: **0** (setelah 5 rekaman dikeluarkan).
  - Ada 18 md5 lintas fold di dalam data latih. Ini dibiarkan dan dicatat sebagai keterbatasan validasi silang.
- **Aturan yang dipegang:**
  - Normalisasi dan augmentasi hanya dari data latih.
  - Hiperparameter dipilih di validasi silang.
  - Data uji dipakai sekali per model per sifat.
  - Seed dan threshold tidak dipilih berdasarkan skor uji.
  - Ada tes otomatis `tests/test_no_leakage.py`.
- **Leave-item-group-out (LIGO):**
  - Alasan: label ditentukan oleh (item, harakat, benar/salah), jadi model bisa saja menghafal item, bukan belajar sifat artikulasinya.
  - 36 item dibagi ke 6 kelompok; varian ikut huruf induknya. Pembagiannya ada di `data/splits/ligo_items.csv`.
  - Tiap lipatan: latih pada data latih tanpa kelompok item itu, lalu uji pada responden uji untuk kelompok item itu.
  - Hasilnya dilaporkan **terpisah** dari skor utama.

## 5. Pra-pemrosesan (sama untuk kedua model)

1. Mono, lalu resample ke 16 kHz (soxr_hq).
2. Trim hening berbasis energi: frame RMS 25 ms / geser 10 ms, ambang 40 dB di bawah puncak, margin 50 ms.
3. Normalisasi amplitudo puncak ke 0,95.
4. Panjang tetap **1,9 s (30.400 sampel)**.
   - Angka ini dari persentil 99 durasi data latih setelah trim, yaitu 1,870 s, dibulatkan ke atas.
   - Rekaman lebih pendek di-*pad* nol di kanan. Rekaman lebih panjang dipotong di tengah; ini terjadi pada 157 rekaman (0,73%).

## 6. Model

### 6.1 Baseline: MFCC + XGBoost

- **Fitur:** 13 MFCC, n_fft 512, jendela 25 ms, geser 10 ms, 40 mel, 20–8000 Hz, ditambah delta dan delta-delta (lebar 9).
  - Diringkas jadi mean + std per koefisien: **78 fitur**.
  - Dihitung hanya pada audio valid (tanpa padding), supaya durasi tidak bocor sebagai fitur.
- **XGBoost** (hist, CPU): subsample 0,8, colsample_bytree 0,8.
  - Grid: n_estimators {300, 600}, max_depth {4, 6}, learning_rate {0,05; 0,1}.
  - Grid dipilih dari rata-rata macro-F1 CV fold 1–5. Terpilih: S1 (600, 6, 0,05); S3 (600, 6, 0,1); T2 (600, 6, 0,05).
  - Lalu dilatih ulang pada seluruh data latih dan diuji sekali di fold 0. Seed 42.

### 6.2 Model usulan: Wav2Vec2-BERT (fine-tune seluruh lapisan)

- **Checkpoint:** `facebook/w2v-bert-2.0`, revisi `da985ba`.
  - Encoder **24 lapisan Conformer**, hidden 1024, 16 *attention head*, FFN 4096, konvolusi *depthwise* kernel 31, posisi relatif.
  - Sekitar **580 juta parameter**.
  - Dilatih awal secara *self-supervised* pada sekitar 4,5 juta jam audio multibahasa (bagian dari Seamless; cek sitasi).
- **Masukan:** sinyal 16 kHz. Fitur log-mel filterbank 80 bin dihitung oleh *feature extractor* bawaan checkpoint (SeamlessM4T), lalu digabung per 2 frame menjadi 160 dimensi × 94 frame, beserta *attention mask*.
- **Kepala klasifikasi:** *masked mean pooling*, dropout 0,1, lalu linear 1024 → 2. Loss *cross-entropy* tanpa class weight.
- **Pelatihan:**
  - AdamW 8-bit, *weight decay* 0,01.
  - Laju belajar backbone 2e-5 dan kepala 1e-3.
  - Batch 8 × akumulasi gradien 4 (batch efektif 32).
  - bf16, *gradient checkpointing*, *gradient clipping* 1,0.
  - *LayerDrop* dan SpecAugment dimatikan.
  - Tanpa augmentasi.
- **Early stopping (protokol C):**
  - Di CV, maksimum 6 epoch dengan *patience* 2 pada macro-F1 validasi; bobot epoch terbaik yang dipakai.
  - Model untuk uji dan LIGO dilatih ulang pada seluruh data latih selama **median epoch terbaik CV**. S1 = 3, S3 = 3; T2 menunggu.
  - Dengan cara ini data uji tidak pernah dipakai untuk memilih model.
- **Seed:** CV dan LIGO memakai seed 42. Uji akhir memakai seed 42, 43, dan 44 untuk melihat variasi antar-seed.
- **Perangkat:** satu laptop RTX 4060 Laptop 8 GB, RAM 24 GB, Windows.
  - Sekitar 18–36 menit per epoch.
  - Total run Wav2Vec2-BERT sekitar 3 hari: 42 unit = 3 sifat × (5 CV + 3 uji + 6 LIGO).
- **Alasan memilih Wav2Vec2-BERT:**
  - Pada shared task Iqra'Eval 2025 (deteksi salah lafal ayat Al-Qur'an), sistem terbaik memakai Wav2Vec2-BERT (F1 0,4726), disusul XLS-R-1B (0,4650).
  - Itu tugas yang berbeda, jadi ini argumen pemilihan model, bukan jaminan hasil.
  - Menurut kajian pustaka yang ada, Wav2Vec2-BERT belum pernah diuji pada klasifikasi sifat huruf hijaiyah.

## 7. Metrik evaluasi

- **Metrik utama:** macro-F1. Juga dilaporkan *accuracy*, *precision*, *recall*, F1, dan AUC-ROC.
- Skor dihitung terpisah untuk subset Data Benar dan Data Salah.
- CV fold 1–5 dilaporkan sebagai mean ± std. Peningkatan dihitung per fold dan pada data uji.
- **Analisis lanjutan:** confusion matrix 2×2 per sifat per model, tingkat kesalahan per item dan per harakat, dan item yang paling sering salah.
- Ada baseline kelas mayoritas sebagai cek kewajaran (macro-F1 uji sekitar 0,33).

## 8. Hasil sejauh ini

### 8.1 XGBoost (final)

| Sifat | CV macro-F1 (mean ± std) | Uji macro-F1 | LIGO macro-F1 (gabungan) |
|---|---|---|---|
| S1 | 0,680 ± 0,022 | 0,688 | 0,624 |
| S3 | 0,652 ± 0,036 | 0,668 | 0,617 |
| T2 | 0,703 ± 0,048 | 0,662 | 0,611 |

### 8.2 Wav2Vec2-BERT (SEMENTARA, run masih berjalan, 26 dari 42 unit)

| Sifat | CV macro-F1 (seed 42) | Uji macro-F1 per seed | LIGO (gabungan, seed 42) |
|---|---|---|---|
| S1 | 0,811 ± 0,085 (fold: 0,687 / 0,767 / 0,901 / 0,840 / 0,861) | 0,798 / 0,801 / 0,766 → rata-rata 0,789 | 0,722 |
| S3 | 0,780 ± 0,085 | 0,758 (seed 42); seed 43 dan 44 [menunggu] | 0,673 |
| T2 | [menunggu] | [menunggu] | [menunggu] |

- **Gambaran awal** (belum final; angka resmi akan keluar dari `reports/improvement.csv`): pada data uji, peningkatan sekitar +0,10 untuk S1 dan sekitar +0,09 untuk S3.
- **LIGO** lebih rendah daripada skor uji pada kedua model. Penurunan S3 lebih besar daripada S1, jadi generalisasi ke huruf baru lebih sulit untuk isti'la.
- **Catatan jujur:**
  - Skor CV Wav2Vec2-BERT sedikit optimistis, karena epoch terbaik dipilih pada fold validasi yang sama. Skor uji fold 0 tidak punya masalah ini, jadi skor uji adalah angka utama.
  - Variasi antar-seed cukup terasa. Contohnya S1 seed 44, yang skor Data Benar-nya 0,685, sementara seed lain di atas 0,81.
  - Fold 1 konsisten paling sulit untuk kedua model.

## 9. Yang belum selesai

- Sisa run Wav2Vec2-BERT (uji S3 seed 43–44 dan seluruh T2). Perkiraan selesai Selasa, 17 November 2026, sore.
- `reports/improvement.csv`, tabel perbandingan akhir, confusion matrix, dan tabel kesalahan per item/harakat untuk Wav2Vec2-BERT.
- **Ablasi augmentasi** (dengan vs tanpa) belum dikerjakan. Rencana augmentasinya: time stretch 0,9–1,1; pitch shift 1–2 semiton; noise SNR sedang; time shift sekitar 100 ms. Keputusannya menunggu pembimbing: dikerjakan penuh, dikerjakan terbatas, atau ditulis sebagai keterbatasan.
- Pertanyaan terbuka ke penulis dataset:
  - Arti akhiran varian `i`, `ie`, `ik`, dan `t`.
  - Apakah responden benar-N dan salah-N adalah orang yang sama. Untuk sekarang split bawaan dipercaya; asumsi ini dicatat di `reports/data_audit.md`.

## 10. Keterbatasan (untuk Bab 1/3/5)

- Label ditentukan per item, sehingga kemampuan generalisasi ke huruf baru hanya dibuktikan lewat LIGO, yang skornya lebih rendah.
- Rekaman sangat pendek dan sangat panjang didominasi Data Salah, sehingga durasi berpotensi menjadi jalan pintas. Untuk memantaunya, skor dilaporkan per subset benar/salah, dan MFCC dihitung tanpa padding.
- Ada 18 rekaman identik lintas fold di data latih, yang membuat CV sedikit optimistis.
- Keterbatasan GPU 8 GB memaksa protokol yang lebih hemat: CV dan LIGO hanya satu seed, dan maksimum 6 epoch.
- Hanya tiga dari dua belas variabel sifat yang dimodelkan, karena konvensi label Data Salah untuk sifat lain tidak tersedia.

## 11. Saran kerangka proposal

- **Bab 1 Pendahuluan:** latar belakang (pentingnya makharij dan sifat huruf dalam tajwid; penilaian manual oleh guru terbatas), rumusan masalah (§1), tujuan, batasan (§1 di luar lingkup, §10), manfaat.
- **Bab 2 Kajian pustaka:**
  - Sifat huruf (Al-Hafiz, 2014).
  - Penelitian sebelumnya pada dataset ini (Azis & Lestari, 2023; dibahas, tidak dibandingkan angka).
  - MFCC.
  - XGBoost (Chen & Guestrin, 2016).
  - *Self-supervised speech*: wav2vec 2.0 (Baevski dkk., 2020), w2v-BERT (Chung dkk., 2021), Conformer (Gulati dkk., 2020), Seamless (2023).
  - Iqra'Eval 2025.
  - *Fine-tuning* hemat memori: AdamW 8-bit (Dettmers dkk., 2022).
  - Semua sitasi wajib dicek.
- **Bab 3 Metodologi:** urutan §2 → §5 → §4 → §6 → §7. Sertakan diagram alur: dataset → praproses → (MFCC → XGBoost) dan (fbank → Wav2Vec2-BERT) → evaluasi (CV, uji, LIGO) → analisis peningkatan.
- **Bab 4 Hasil:** `[menunggu hasil final]`. Struktur: tabel per sifat per model, kolom peningkatan, subset benar/salah, LIGO, confusion matrix, dan analisis per item/harakat.
