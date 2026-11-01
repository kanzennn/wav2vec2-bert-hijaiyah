# CLAUDE.md — TA: Klasifikasi Sifat Huruf Hijaiyah (versi 4, berdiri sendiri)

Panduan untuk Claude Code di repo ini. Baca seluruhnya sebelum mengubah kode. Dokumen ini menggantikan semua versi CLAUDE.md sebelumnya; abaikan instruksi lama jika ada di riwayat atau berkas lain. Pemilik: Faiz Elfahad Kurniawan (Informatika, Telkom University). Balas dan beri komentar penjelasan dalam Bahasa Indonesia informal; nama variabel, fungsi, dan commit message dalam bahasa Inggris.

## 0. Konteks dan hal yang berubah

TA ini membandingkan enam model untuk **klasifikasi sifat huruf (sifat al-huruf) hijaiyah** dari rekaman pelafalan pendek. Repo ini mungkin masih berisi hasil lama dari konsep **identifikasi huruf (36 kelas item)**. Konsep itu sudah ditinggalkan:

- Label target sekarang adalah **sifat huruf** (biner), bukan nomor item.
- Yang dipakai adalah **Data Benar dan Data Salah**, karena kontras benar-salah itulah yang membuat sifat dapat diklasifikasi.
- Hasil lama jangan dihapus. Pindahkan ke `reports/archive_letter_id/` dan `runs/archive_letter_id/` dengan `README.md` singkat (tanggal dan konfigurasi). Jangan dicampur dengan hasil baru di `reports/results.csv`.
- Kode praproses, ekstraksi fitur, dan model backbone boleh dipakai ulang. Yang diganti: pembacaan label, kepala klasifikasi, loss, metrik, split, dan laporan.

Sebelum mengubah apa pun, lakukan langkah 7.1 (inventarisasi) dan tunjukkan rencana ke user.

## 1. Tujuan proyek

| Kelompok | Model | Masukan |
|---|---|---|
| Klasik | SVM (RBF), Random Forest, XGBoost | ringkasan MFCC (mean + std per koefisien) |
| Deep learning | CNN, CRNN | Mel spectrogram (log-mel, dB) |
| Pretrained | wav2vec 2.0 (mode beku dan fine-tune) | sinyal mentah 16 kHz |

Pertanyaan penelitian: (1) model mana yang paling baik mengklasifikasi tiap sifat pada responden yang tidak pernah dilihat, (2) seberapa besar pengaruh jenis fitur (ablasi MFCC vs Mel spectrogram), (3) sifat dan kondisi mana yang paling sulit (per sifat, per item, per harakat).

Di luar lingkup: perangkat keras ESP32 dan mikrofon, identifikasi huruf 36 kelas, dan pembuatan aplikasi.

## 2. Dataset

Hijaiyah Letter Pronunciation Audio Dataset (A. M. H. Azis, Mendeley Data V1, doi 10.17632/r5fy2dd98g.1, lisensi CC BY 4.0). Data sudah ada di komputer user sebagai satu berkas **`Hijaiyah Letter Pronunciation Audio Dataset.zip`**. Jangan mengunduh dari internet; **ekstrak sendiri** berkas itu.

Langkah ekstraksi (lakukan di awal, sebelum inventarisasi 7.1):
1. Cari berkas zip itu di root repo atau di folder induknya (`find . .. -maxdepth 2 -name "Hijaiyah Letter Pronunciation Audio Dataset.zip"`). Bila tidak ketemu, tanya user lokasinya.
2. Lihat isinya dulu (`unzip -l`) dan pastikan tidak ada path berbahaya (diawali `/` atau mengandung `..`). Ekstrak ke folder kosong baru, mis. `data/raw_extract/`, jangan menimpa folder lain.
3. Zip ini berisi (menurut README dataset) `Data_Benar.zip`, `Data_Salah.zip`, `metadata_rekaman.csv`, `label_per_huruf.csv`, `rekaman_dikeluarkan.csv`, `Kode_Pipeline.zip`, dan README. Ekstrak zip bersarang `Data_Benar.zip` dan `Data_Salah.zip` juga. Bila struktur sebenarnya berbeda, laporkan ke user sebelum memindahkan apa pun.
4. Susun hasilnya ke tata letak berikut, lalu pindahkan atau hapus `data/raw_extract/` setelah verifikasi:

```
data/raw/Data Benar/1 ... 20/*.wav
data/raw/Data Salah/1 ... 20/*.wav
data/labels/metadata_rekaman.csv
data/labels/label_per_huruf.csv
data/labels/rekaman_dikeluarkan.csv
```

5. Verifikasi: jumlah WAV Data Benar 10.800 berkas di disk (10.791 di metadata, 9 berkas `Data Benar/12/4-100.wav` sampai `4-108.wav` dikeluarkan) dan Data Salah 10.800; metadata 21.591 baris. Laporkan selisih apa pun ke user.
6. **`Kode_Pipeline.zip` hanya boleh dibaca sebagai acuan** (ekstrak ke `data/raw_extract/Kode_Pipeline/`, di luar `src/`). Jangan menjalankan kodenya dan jangan menyalinnya mentah.
7. Data mentah tidak boleh di-commit; tambahkan `data/raw/`, `data/raw_extract/`, dan `*.zip` ke `.gitignore`.

Bila ada berkas yang hilang setelah ekstraksi, berhenti dan tanya user. Jangan menebak.

### 2.1 Fakta dataset

- Mono WAV, 44,1 kHz (3 berkas 48 kHz). Data Benar 10.791 rekaman, Data Salah 10.800, total 21.591 (9 berkas bernama tidak standar di `Data Benar/12`, `4-100.wav` sampai `4-108.wav`, dikeluarkan dataset).
- Tiap kelas 20 folder responden; 540 berkas per responden (36 item x 3 harakat x 5 pengulangan). Harakat: 1 fathah, 2 kasrah, 3 dhammah.
- Nama berkas: `<item>-<harakat>-<pengulangan>.wav`, contoh `21-1-1.wav` (item 21 = ق, fathah, ulangan 1) dan `25i-1-1.wav`. Parse dengan regex `^(\d+)([A-Za-z]*)-(\d)-(\d+)\.wav$` menjadi `huruf_no`, `varian`, `harakat`, `pengulangan`.
- **Nomor huruf pada nama berkas sama dengan `no_huruf` di `label_per_huruf.csv`** (dikonfirmasi user).
- **36 item** = 27 item biasa (1-24, 26, 27, 28) + tiga varian huruf 25 (`25i`, `25ie`, `25ik`; tidak ada item `25` polos) + enam varian `t` (`2t`, `5t`, `8t`, `16t`, `21t`, `26t`). **Arti akhiran `i`, `ie`, `ik`, `t` belum dijelaskan dataset; jangan ditebak** (user akan menanyakannya ke Pak Aaz). Perlakukan varian sebagai item biasa dengan label dari metadata.
- Per item 300 rekaman per kategori, kecuali `26t`, `27`, `28` di Data Benar yang 297 (99 per harakat).
- Durasi **tidak seragam**: median sekitar 0,83 detik; persentil 1 dan 99 = 0,23 dan 1,82 detik; minimum 0,03 detik, maksimum 11,1 detik; 110 rekaman lebih dari 2 detik dan 754 kurang dari 0,3 detik. README yang menyebut 0,7 sampai 1,0 detik hanya perkiraan kasar.
- 26 berkas `*-error` di `Data Benar/15` (`flag_error=True`) diperlakukan sebagai pengucapan benar.
- Beberapa folder responden berisi rekaman identik: Data Benar 8, 18, 19 dan Data Salah 13, 15. Kolom `grup_cv` menggabungkannya (`benar-8_18_19`, `salah-13_15`).

### 2.2 `metadata_rekaman.csv` (sumber utama)

21.591 baris, 20 kolom: `source_id, category (benar/salah), folder, filename, responden, grup_cv, item, huruf_no, sub (harakat 1-3), rep, flag_error, sr, duration, md5, y_S1, y_S3, y_T2, split (latih/uji), fold (0-5), kode_uji`.

- **Label training = kolom `y_S1`, `y_S3`, `y_T2`.** Label Data Salah sudah dibalik oleh dataset (kebalikan label Data Benar pada item yang sama). Jangan membalik lagi.
- **Split dan fold bawaan dipakai apa adanya**; jangan membuat split sendiri:
  - `split=uji` (fold 0): 4.311 rekaman = responden benar-2, 9, 12, 13 dan salah-2, 4, 10, 11 (`kode_uji` B1-B4, K1-K4).
  - `split=latih`: 17.280 rekaman di fold 1 sampai 5 menurut `grup_cv` (6, 6, 6, 5, 6 kelompok per fold). Satu `grup_cv` tidak boleh terpecah antar fold.
- Sebaran label per rekaman (sudah dihitung):

| Sifat | Data Benar kelas 0 | Data Benar kelas 1 | Gabungan 0 / 1 |
|---|---|---|---|
| S1 (jahr 0 / hams 1) | 7.794 | 2.997 | 10.794 / 10.797 |
| S3 (istifal 0 / isti'la 1) | 8.091 | 2.700 | 10.791 / 10.800 |
| T2 (tanpa 0 / qalqalah 1) | 7.791 | 3.000 | 10.791 / 10.800 |

  Gabungan hampir 50:50. **Verifikasi angka ini di data audit**; pakai class weight hanya bila ternyata tidak seimbang di data latih.
- Label S1 dan T2 mengikuti huruf. **S3 juga bergantung harakat** untuk huruf 7 (خ), 10 (ر), 19 (غ): tafkhim kecuali dengan kasrah. Karena itu S3 per rekaman berbeda dari S3 per huruf di `label_per_huruf.csv` untuk huruf-huruf ini; **metadata yang benar**.
- Label varian (Data Benar): `25i`, `25ie`, `25ik`, `26t` semuanya 0 untuk S1/S3/T2; `2t`, `5t`, `8t` T2=1 (S1=0, S3=0); `16t` dan `21t` S3=1 dan T2=1 (S1=0).

### 2.3 `label_per_huruf.csv`

28 baris, kolom `no_huruf,huruf,S1,S2,S3,S4,S5,T1,T2,T3,T4,T5,T6,T7` (baca dengan `encoding="utf-8-sig"`). Dipakai untuk referensi, tabel laporan, dan **tes konsistensi**: untuk huruf 1 sampai 28 label S1 dan T2 di metadata harus sama dengan CSV, dan S3 sama kecuali huruf 7, 10, 19. Jumlah huruf kelas 1: S1 hams 10 huruf, S3 isti'la 7 huruf (خ ص ض ط ظ غ ق), T2 qalqalah 5 huruf (ب ج د ط ق).

### 2.4 Skema label menurut dataset

Dua belas variabel mencakup 18 sifat (rujukan Al-Hafiz, 2014): S1 jahr/hams, S2 rakhawah/bayniyyah/syiddah (tiga nilai), S3 istifal/isti'la, S4 infitah/ithbaq, S5 ishmat/idzlaq; T1 shafir, T2 qalqalah, T3 liin, T4 inhirah, T5 takrir, T6 tafasysyi, T7 istithaalah (1 ada, 0 tidak). Artikel penulis dataset hanya memodelkan S1, S3, T2. Konvensi label Data Salah untuk sembilan sifat lainnya belum terisi; metadata hanya menyediakan S1, S3, T2.

### 2.5 Yang belum pasti (tanya user, jangan menebak)

- Arti akhiran varian (`i`, `ie`, `ik`, `t`).
- Apakah responden `benar-N` dan `salah-N` orang yang sama. Dataset memisahkan keduanya di split (uji benar-2, 9, 12, 13 dan salah-2, 4, 10, 11), jadi percayai split bawaan, tetapi catat asumsi ini di `reports/data_audit.md`.

## 3. Target klasifikasi

Tahap utama (wajib), sama dengan artikel dataset:

| ID tugas | Sifat | Label | Kelas |
|---|---|---|---|
| `S1` | jahr vs hams | `y_S1` | 2 |
| `S3` | istifal vs isti'la | `y_S3` | 2 |
| `T2` | qalqalah ada vs tidak | `y_T2` | 2 |

Tahap tambahan (opsional, hanya setelah tahap utama selesai dan user setuju): sifat lain hanya bila label dan konvensi Data Salah-nya jelas dan tersedia. Catat alasan bila dilewati.

- Satu model dilatih **per sifat**. Untuk CNN, CRNN, dan wav2vec 2.0 boleh ada varian multi-task tiga kepala, tapi baseline yang dilaporkan tetap per sifat dan keduanya dilaporkan terpisah.
- Data: seluruh 21.591 rekaman. Tambahkan kolom `is_correct` (1 = benar, 0 = salah) di indeks dan simpan di prediksi untuk analisis, tetapi **jangan** jadikan `is_correct` sebagai fitur.
- Tulis `get_label(task, row)` yang hanya membaca kolom `y_*` dari metadata, dengan unit test konsistensi terhadap `label_per_huruf.csv` (lihat 2.3).

## 4. Aturan anti-leakage (WAJIB)

1. Pakai split dan fold dari metadata (speaker-exclusive menurut dataset). Satu `grup_cv` tidak boleh ada di dua split atau dua fold.
2. Verifikasi dengan kolom `md5` dan hash isi berkas bahwa tidak ada isi sama di data latih dan uji; laporkan ke `reports/data_audit.md`.
3. Augmentasi hanya pada data latih, setelah split. Salinan augmentasi ikut split dan fold sumbernya.
4. Normalisasi fitur hanya dari data latih (atau latih fold terkait).
5. Hiperparameter dipilih di validasi (rotasi fold 1 sampai 5). Data uji (fold 0) dipakai sekali per model dan sifat.
6. Jangan memilih model, seed, atau threshold berdasarkan skor uji.
7. **Kebocoran lewat identitas item:** label ditentukan oleh (item, harakat, benar/salah), jadi model bisa menghafal item atau status benar/salah, bukan sifat artikulasinya. Wajib jalankan evaluasi **leave-item-group-out** (kelompok item ditahan sebagai uji, misalnya 6 lipatan, tiap lipatan berisi huruf dari beberapa kelas sifat dan tidak semua huruf qalqalah di satu lipatan). Laporkan terpisah dari skor utama dan jelaskan perbedaannya. Simpan lipatannya di `data/splits/`.

Tambahkan `tests/test_no_leakage.py`: gagal bila ada `grup_cv` di dua split atau fold, md5 sama lintas split latih-uji, atau `is_correct` ikut masuk ke fitur.

## 5. Pipeline (urutan = Bab 3 proposal)

1. **Pengumpulan data:** indeks 21.591 berkas, cocokkan dengan metadata (berkas hilang atau berlebih dilaporkan), audit kualitas (rusak, kosong, durasi ekstrem). Buat `reports/outlier_durasi.csv` untuk rekaman di bawah 0,3 detik dan di atas 2 detik; **jangan membuang tanpa persetujuan user**.
2. **Pra-pemrosesan (sama untuk semua model):** mono, resample 16 kHz, trim hening dengan ambang energi, normalisasi amplitudo, lalu pad atau potong ke panjang tetap **sekitar 2,0 detik (32.000 sampel)**. Putuskan nilai akhirnya dari distribusi durasi data latih setelah trim (persentil 99), catat di `reports/data_audit.md`.
3. **Ekstraksi fitur:** MFCC (jendela 25 ms, geser 10 ms, statis + delta + delta-delta, diringkas mean + std) untuk model klasik; Mel spectrogram (sekitar 64 pita, log/dB) untuk CNN dan CRNN; sinyal mentah 16 kHz untuk wav2vec 2.0.
4. **Pembagian data:** baca split dan fold dari metadata, simpan ke `data/splits/*.csv` bersama lipatan leave-item-group-out.
5. **Augmentasi (hanya latih):** time stretch 0,9 sampai 1,1; pitch shift 1 sampai 2 semiton; noise SNR sedang; time shift sekitar 100 ms; SpecAugment untuk model Mel. Peringatan: pitch shift dan time stretch berlebihan dapat mengubah sifat artikulasi (misalnya jahr/hams). Uji lewat ablasi (dengan vs tanpa) di validasi dan nonaktifkan yang menurunkan skor.
6. **Pelatihan:** SVM RBF, Random Forest, XGBoost, CNN (blok conv-BN-ReLU-pool), CRNN (CNN + 1 sampai 2 lapis GRU/LSTM), wav2vec 2.0 beku dan fine-tune. Loss cross-entropy (class weight bila perlu). Deep learning: Adam, early stopping di validasi.
7. **Pengujian dan evaluasi (per sifat per model):** accuracy, precision, recall, F1, **macro-F1 sebagai metrik utama**, AUC-ROC. Skor terpisah untuk subset Data Benar dan Data Salah. Validasi silang 5-fold dari fold 1 sampai 5, laporkan mean +- std. Ablasi MFCC vs Mel pada model yang bisa menerima keduanya.
8. **Analisis hasil:** confusion matrix 2x2 per sifat per model, tingkat kesalahan per item dan per harakat (termasuk varian), item yang paling sering salah, dan tabel model terbaik per sifat.

Cek kewajaran: bila baseline sederhana (kelas mayoritas) mendekati model Anda, atau skor mendekati 100% pada split per responden, curigai kebocoran atau label trivial dan periksa dulu sebelum melapor.

## 6. Struktur repo

```
configs/            # satu YAML per eksperimen (sifat, fitur, model, hiperparameter, seed)
data/raw/           # dataset asli (read-only, jangan commit)
data/labels/        # metadata_rekaman.csv, label_per_huruf.csv, rekaman_dikeluarkan.csv
data/interim/       # audio terproses dan cache fitur
data/splits/        # split latih/uji + fold 1-5 + leave-item-group-out
src/ta/
  data/             # index, audit, labels (get_label), preprocess, split, augment
  features/         # mfcc.py, melspec.py, raw.py
  models/           # classical.py, cnn.py, crnn.py, wav2vec.py
  train.py          # python -m ta.train --config configs/xxx.yaml --task S1
  evaluate.py       # metrik, confusion matrix, ablasi, analisis per item/harakat
tests/
reports/            # results.csv, gambar, data_audit.md, label_inventory.md, CHANGELOG.md
reports/archive_letter_id/
runs/
runs/archive_letter_id/
```

## 7. Cara kerja yang diharapkan dari Claude Code

1. **Ekstrak data dulu** (bagian 2), lalu **inventarisasi.** Sebelum menulis kode baru: (a) baca struktur repo dan hasil lama, (b) baca ketiga berkas di `data/labels/` dan cocokkan dengan bagian 2 (jumlah baris, kolom, sebaran label), (c) tulis ringkasan di `reports/label_inventory.md`, (d) tampilkan rencana perubahan ke user dan tunggu konfirmasi. Hentikan dan tanya bila ada yang berbeda dari bagian 2.
2. Arsipkan hasil lama (bagian 0), lalu tulis `get_label` dan unit test-nya.
3. Jalankan data audit baru; hentikan dan tanya bila jumlah menyimpang (harus 10.791 + 10.800 = 21.591).
4. Bangun baseline termurah dulu: MFCC + SVM untuk S1, S3, T2 end-to-end, lolos test anti-leakage. Baru lanjut ke model lain, satu kelompok per langkah (klasik, CNN/CRNN, wav2vec 2.0). Jalankan `--smoke` sebelum training penuh.
5. Setelah tiap run, tambahkan satu baris ke `reports/results.csv`: sifat, model, fitur, mode, seed, fold, accuracy, macro-F1, precision, recall, F1, AUC. Simpan prediksi (dengan `item`, `harakat`, `is_correct`) dan confusion matrix.
6. Reproducibility: satu `seed` di config untuk python, numpy, torch; model deep learning dari minimal 3 seed. Tiap run menyimpan salinan config, metrik, prediksi, dan git commit hash. Cache fitur ke disk. Deteksi GPU otomatis.
7. Jangan mengarang angka; semua angka laporan berasal dari `runs/`. Eksperimen yang gagal atau tidak dijalankan ditulis apa adanya.
8. Jangan mengunduh model atau data dari internet tanpa izin. Untuk wav2vec 2.0, tanya user checkpoint-nya (mis. `facebook/wav2vec2-base`) dan catat versinya.
9. Commit kecil dan sering, pesan dalam bahasa Inggris. Jangan commit `data/raw/`, checkpoint, atau berkas lebih dari 50 MB.

## 8. Keluaran untuk laporan TA

- `reports/results.csv` dan tabel perbandingan enam model per sifat (mean +- std antar-fold).
- `reports/ablation_features.csv` (MFCC vs Mel) dan ablasi augmentasi.
- Confusion matrix 2x2 per sifat per model, plus tabel kesalahan per item dan per harakat.
- Hasil leave-item-group-out, terpisah dari skor utama.
- `reports/label_inventory.md`, `reports/data_audit.md`, `reports/outlier_durasi.csv`, `reports/CHANGELOG.md`.
- Ringkasan akhir 1 halaman: model terbaik per sifat, alasan (angka), keterbatasan, saran lanjutan.

## 9. Catatan jujur untuk proposal

- Nilai parameter di bagian 5 adalah rencana awal, bukan hasil; ubah bila validasi menunjukkan lain dan catat alasannya.
- S1, S3, T2 mengikuti artikel penulis dataset agar dapat dibandingkan.
- Karena label ditentukan per item, kemampuan generalisasi ke item baru harus dibuktikan lewat leave-item-group-out; sebutkan sebagai keterbatasan bila skornya jauh lebih rendah dari skor utama.
