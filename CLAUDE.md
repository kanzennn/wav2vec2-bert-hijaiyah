# CLAUDE.md — TA: Klasifikasi Sifat Huruf Hijaiyah (versi 7, berdiri sendiri)

Panduan untuk Claude Code di repo ini. Baca seluruhnya sebelum mengubah kode. Dokumen ini menggantikan semua versi CLAUDE.md sebelumnya (versi 7: eksperimen lanjutan dijalankan di laptop teman, lihat bagian 10; bila bagian 10 bertentangan dengan bagian 1 sampai 9, bagian 10 yang berlaku); abaikan instruksi lama jika ada di riwayat atau berkas lain. Pemilik: Faiz Elfahad Kurniawan (Informatika, Telkom University). Balas dan beri komentar penjelasan dalam Bahasa Indonesia informal; nama variabel, fungsi, dan commit message dalam bahasa Inggris.

## 0. Konteks dan hal yang berubah

TA ini berjudul **"Analisis Peningkatan Klasifikasi Sifat Huruf Hijaiyah Menggunakan Model Pretrained Self-Supervised"**. TA ini mengukur seberapa besar **peningkatan** kinerja **klasifikasi sifat huruf (sifat al-huruf) hijaiyah** dari rekaman pelafalan pendek ketika memakai model pretrained Wav2Vec2-BERT yang di-fine-tune dibandingkan baseline XGBoost dengan fitur MFCC. Hanya **dua model** yang dipakai; cakupan sudah sengaja dikecilkan atas arahan dosen pembimbing. Jangan menambah model lain (SVM, Random Forest, CNN, CRNN, wav2vec 2.0 beku) tanpa persetujuan user. Pengecualian yang sudah disetujui (v6, lihat bagian 10): pembanding **Wav2Vec2-BERT beku + probe linear** (checkpoint yang sama) dan **XGBoost dengan grid diperluas + 3 seed**. Repo ini mungkin masih berisi hasil lama dari konsep **identifikasi huruf (36 kelas item)**. Konsep itu sudah ditinggalkan:

- Label target sekarang adalah **sifat huruf** (biner), bukan nomor item.
- Yang dipakai adalah **Data Benar dan Data Salah**, karena kontras benar-salah itulah yang membuat sifat dapat diklasifikasi.
- Hasil lama jangan dihapus. Pindahkan ke `reports/archive_letter_id/` dan `runs/archive_letter_id/` dengan `README.md` singkat (tanggal dan konfigurasi). Jangan dicampur dengan hasil baru di `reports/results.csv`.
- Kode praproses, ekstraksi fitur, dan model backbone boleh dipakai ulang. Yang diganti: pembacaan label, kepala klasifikasi, loss, metrik, split, dan laporan.

Sebelum mengubah apa pun, lakukan langkah 7.1 (inventarisasi) dan tunjukkan rencana ke user.

## 1. Tujuan proyek

| Peran | Model | Masukan |
|---|---|---|
| Baseline | XGBoost | ringkasan MFCC (mean + std per koefisien, termasuk delta dan delta-delta) |
| Model usulan | Wav2Vec2-BERT, fine-tune seluruh lapisan, kepala klasifikasi dua kelas | sinyal 16 kHz; log-mel filterbank dihitung oleh feature extractor bawaan checkpoint (jangan dihitung manual) |

Cadangan bila GPU tidak cukup: XLS-R-300M (`facebook/wav2vec2-xls-r-300m`). Jangan beralih ke cadangan tanpa memberi tahu user dan mencatatnya di `reports/CHANGELOG.md`. Checkpoint Wav2Vec2-BERT yang dipakai (misalnya `facebook/w2v-bert-2.0`) harus dikonfirmasi user lebih dulu.

Rumusan masalah dan tujuan (proposal v18, hanya dua): (1) membangun model klasifikasi sifat huruf hijaiyah S1, S3, dan T2 berbasis Wav2Vec2-BERT yang di-fine-tune dengan pembagian per responden; (2) mengukur pada sifat apa saja model itu meningkatkan macro-F1 dibandingkan XGBoost + MFCC, dengan selang kepercayaan 95% tingkat responden. Analisis galat per huruf, pembanding encoder beku, LIGO, dan biaya komputasi adalah analisis pendukung.

**Peningkatan** = macro-F1 Wav2Vec2-BERT dikurangi macro-F1 XGBoost, pada data uji dan fold yang sama, per sifat. **XGBoost harus dilatih ulang pada split ini.** Jangan membandingkan langsung dengan angka penelitian Azis & Lestari (2023): mereka memakai 12 label, akurasi sebagai metrik, dan split 15+15 latih / 5+5 uji, sedangkan TA ini memakai tiga sifat, macro-F1, dan split bawaan dataset 16+16 / 4+4.

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
- **36 item** = 27 item biasa (1-24, 26, 27, 28) + tiga varian huruf 25 (`25i`, `25ie`, `25ik`; tidak ada item `25` polos) + enam varian `t` (`2t`, `5t`, `8t`, `16t`, `21t`, `26t`). **Arti akhiran `i`, `ie`, `ik`, `t` belum dijelaskan dataset; jangan ditebak** (user akan menanyakannya ke Pak Aaz). **Keputusan terbaru (proposal v16): varian digabung ke huruf induknya menurut nomor huruf (kolom `huruf_no`; 2t,5t,8t,16t,21t,26t,25i,25ie,25ik masuk ke huruf 2,5,8,16,21,26,25), sehingga analisis per item memakai 28 huruf.** Label varian sudah dicek identik dengan huruf induknya (Data Benar dan Data Salah). Analisis galat per item (`galat_per_item.csv`) harus dihitung ulang dengan `huruf_no` sebagai unit; angka lama (36 item) jangan dikutip di proposal.
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
- Sudah diverifikasi dari metadata: tiap `responden` hanya ada di satu fold dan satu split (irisan responden latih dan uji kosong); 40 responden menjadi 37 `grup_cv` (`benar-8_18_19` dan `salah-13_15` digabung). Ulangi verifikasi ini di data audit.

## 3. Target klasifikasi

Tahap utama (wajib), sama dengan artikel dataset:

| ID tugas | Sifat | Label | Kelas |
|---|---|---|---|
| `S1` | jahr vs hams | `y_S1` | 2 |
| `S3` | istifal vs isti'la | `y_S3` | 2 |
| `T2` | qalqalah ada vs tidak | `y_T2` | 2 |

Tahap tambahan (opsional, hanya setelah tahap utama selesai dan user setuju): sifat lain hanya bila label dan konvensi Data Salah-nya jelas dan tersedia. Catat alasan bila dilewati.

- Satu model dilatih **per sifat**, untuk XGBoost maupun Wav2Vec2-BERT. Varian multi-task tiga kepala pada Wav2Vec2-BERT hanya boleh dikerjakan bila user menyetujui; baseline yang dilaporkan tetap per sifat dan hasil multi-task dilaporkan terpisah.
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
3. **Ekstraksi fitur:** MFCC (jendela 25 ms, geser 10 ms, statis + delta + delta-delta, diringkas mean + std) untuk XGBoost; sinyal 16 kHz untuk Wav2Vec2-BERT, yang menghitung fitur log-mel filterbank lewat feature extractor bawaan checkpoint.
4. **Pembagian data:** baca split dan fold dari metadata, simpan ke `data/splits/*.csv` bersama lipatan leave-item-group-out.
5. **Augmentasi (hanya latih):** time stretch 0,9 sampai 1,1; pitch shift 1 sampai 2 semiton; noise SNR sedang; time shift sekitar 100 ms; masking rentang waktu opsional saat fine-tune Wav2Vec2-BERT. Peringatan: pitch shift dan time stretch berlebihan dapat mengubah sifat artikulasi (misalnya jahr/hams). Uji lewat ablasi (dengan vs tanpa) di validasi dan nonaktifkan yang menurunkan skor.
6. **Pelatihan:** XGBoost (hiperparameter: laju belajar, kedalaman, jumlah pohon, dipilih di validasi) dan Wav2Vec2-BERT fine-tune seluruh lapisan dengan kepala klasifikasi dua kelas (AdamW, laju belajar kecil, cross-entropy dengan class weight bila perlu, early stopping di validasi).
7. **Pengujian dan evaluasi (per sifat per model):** accuracy, precision, recall, F1, **macro-F1 sebagai metrik utama**, AUC-ROC. Skor terpisah untuk subset Data Benar dan Data Salah. Validasi silang 5-fold dari fold 1 sampai 5, laporkan mean +- std. Hitung peningkatan per sifat per fold (macro-F1 Wav2Vec2-BERT dikurangi macro-F1 XGBoost) dan laporkan mean +- std.
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
  features/         # mfcc.py, raw.py
  models/           # xgboost_model.py, wav2vec_bert.py
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
4. Bangun baseline termurah dulu: MFCC + XGBoost untuk S1, S3, T2 end-to-end, lolos test anti-leakage. Baru lanjut ke Wav2Vec2-BERT. Jalankan `--smoke` sebelum training penuh.
5. Setelah tiap run, tambahkan satu baris ke `reports/results.csv`: sifat, model, fitur, mode, seed, fold, accuracy, macro-F1, precision, recall, F1, AUC. Simpan prediksi (dengan `item`, `harakat`, `is_correct`) dan confusion matrix.
6. Reproducibility: satu `seed` di config untuk python, numpy, torch; Wav2Vec2-BERT dari minimal 3 seed. Tiap run menyimpan salinan config, metrik, prediksi, dan git commit hash. Cache fitur ke disk. Deteksi GPU otomatis.
7. Jangan mengarang angka; semua angka laporan berasal dari `runs/`. Eksperimen yang gagal atau tidak dijalankan ditulis apa adanya.
8. Jangan mengunduh model atau data dari internet tanpa izin. Untuk Wav2Vec2-BERT, tanya user checkpoint-nya (mis. `facebook/w2v-bert-2.0`) dan catat versinya.
9. Commit kecil dan sering, pesan dalam bahasa Inggris. Jangan commit `data/raw/`, checkpoint, atau berkas lebih dari 50 MB.

## 8. Keluaran untuk laporan TA

- `reports/results.csv` dan tabel perbandingan dua model per sifat (mean +- std antar-fold), beserta kolom peningkatan macro-F1.
- Ablasi augmentasi (dengan vs tanpa).
- Confusion matrix 2x2 per sifat per model, plus tabel kesalahan per item dan per harakat.
- Hasil leave-item-group-out, terpisah dari skor utama.
- `reports/label_inventory.md`, `reports/data_audit.md`, `reports/outlier_durasi.csv`, `reports/CHANGELOG.md`.
- Ringkasan akhir 1 halaman: model terbaik per sifat, alasan (angka), keterbatasan, saran lanjutan.

## 9. Catatan jujur untuk proposal

- Nilai parameter di bagian 5 adalah rencana awal, bukan hasil; ubah bila validasi menunjukkan lain dan catat alasannya.
- S1, S3, T2 mengikuti artikel penulis dataset agar dapat dibandingkan.
- Alasan memilih Wav2Vec2-BERT: pada shared task Iqra'Eval 2025 (penilaian pelafalan Al-Qur'an), sistem terbaik memakai Wav2Vec2-BERT (F1 0,4726) dan XLS-R-1B (0,4650). Itu tugas deteksi salah lafal pada ayat, bukan klasifikasi sifat huruf, jadi ini argumen pemilihan, bukan jaminan hasil. Wav2Vec2-BERT belum pernah diuji pada klasifikasi sifat huruf hijaiyah menurut kajian pustaka yang ada.
- Jangan melaporkan peningkatan terhadap angka penelitian lain; hanya terhadap XGBoost yang dilatih ulang pada split yang sama.
- Karena label ditentukan per item, kemampuan generalisasi ke item baru harus dibuktikan lewat leave-item-group-out; sebutkan sebagai keterbatasan bila skornya jauh lebih rendah dari skor utama.

## 10. Eksperimen lanjutan (v7): rancangan di komputer Faiz, eksekusi di laptop teman

Bagian ini menimpa bagian 1 sampai 9 bila bertentangan. Khusus: **augmentasi dan ablasi augmentasi TIDAK dikerjakan** (proposal subbab 3.4.6 mencatatnya sebagai keterbatasan); abaikan langkah augmentasi di bagian 5 dan 8. Model utama (XGBoost 1 seed dan Wav2Vec2-BERT 3 seed) sudah selesai dan **tidak dilatih ulang**.

Tujuan: menyelesaikan eksperimen yang dijanjikan proposal (subbab 3.4.5, 3.5, 3.6) tetapi belum punya hasil. Semua hasil **eksploratif** karena data uji sudah pernah dilihat (proposal subbab 3.1); jangan menulis "signifikan" tanpa "eksploratif" dan jangan menyebutnya konfirmatori. **Jangan mengubah** split, seed uji (42, 43, 44), ambang 0,5, preprocessing, atau metode statistik (10.8). Jangan menyetel apa pun dengan data uji.

### 10.1 Pembagian kerja dan aturan di laptop teman
- Rancangan, penulisan proposal, dan pembacaan hasil dilakukan Faiz di komputernya. Claude Code di laptop teman **hanya menjalankan eksperimen, mencatat, dan menyerahkan hasil**. Jangan mengedit berkas proposal atau menarik kesimpulan akhir; tulis ke `reports/RINGKASAN_LANJUTAN.md` saja.
- Ini laptop milik orang lain. Bekerja **hanya di dalam folder repo**; jangan membuka, mencari, atau mengubah berkas di luarnya. Jangan menyimpan token, kata sandi, atau login akun Faiz maupun akun teman. Jangan `git push` kecuali Faiz meminta, dan bila diminta hanya ke branch `exp-lanjutan`, bukan `main`.
- Jangan mengunduh data atau model dari internet tanpa izin Faiz (paket pip dari PyPI untuk menyiapkan environment boleh). Jangan commit `data/raw/`, `pretrained/`, checkpoint model, atau berkas lebih dari 50 MB.
- Setelah semua selesai, tanya Faiz apakah data mentah, cache, dan checkpoint di laptop teman perlu dihapus; jangan menghapus sebelum ada jawaban.

### 10.2 Barang yang harus ada (Faiz yang membawa; jangan ditebak atau diunduh sendiri)
1. Repo ini beserta perubahan eksperimen lanjutan. Perubahan itu ada di `eksperimen_tambahan.patch` (berisi `src/ta/analysis.py`, `configs/xgb_mfcc_ext.yaml`, `configs/w2v_bert_frozen.yaml`, `freeze_backbone` di `wav2vec_bert.py`, `test_seeds` di `train.py`, tes baru). Bila berkasnya belum ada di repo, jalankan `git apply --check eksperimen_tambahan.patch` lalu `git apply eksperimen_tambahan.patch`; bila gagal, berhenti dan tanya.
2. `Hijaiyah Letter Pronunciation Audio Dataset.zip` (ekstrak sesuai bagian 2).
3. Folder checkpoint `pretrained/facebook-w2v-bert-2.0` (revisi `da985ba`). Config memakai jalur lokal ini.
4. Hasil run utama untuk analisis, cukup berkas prediksi dan metrik (tanpa checkpoint): `runs/20261102-004931_xgb_mfcc_all/` dan `runs/20261109-100209_w2v_bert_all/`.
5. `data/labels/` dan `data/splits/` (agar split identik).

Bila salah satu tidak ada, berhenti dan minta ke Faiz. Jangan membuat ulang split atau mengganti revisi checkpoint.

### 10.3 Pemeriksaan awal laptop (E0), tulis ke `reports/hardware_laptop.md`
1. Catat: OS, CPU (model dan jumlah core), RAM, GPU (`nvidia-smi`: model dan VRAM), disk kosong, versi Python, torch, transformers, xgboost, bitsandbytes, dan `torch.cuda.is_available()`. Berkas ini wajib karena biaya komputasi (10.6) harus diukur di mesin yang sama untuk kedua model.
2. Siapkan environment bersih (venv atau conda) dari `pyproject.toml`. Bila `bitsandbytes` gagal dipasang (umum di Windows atau tanpa GPU NVIDIA), **jangan** mengganti `optimizer: adamw8bit` sendiri; laporkan ke Faiz, karena itu menyimpang dari konfigurasi model utama.
3. Putuskan jalur menurut sumber daya, lalu laporkan ke Faiz sebelum lanjut:
   - VRAM 8 GB atau lebih: semua eksperimen (E1 sampai E6).
   - VRAM kurang dari 8 GB: E3 (probe beku) tetap bisa, turunkan `batch_size` dan naikkan `grad_accum` agar batch efektif tetap 32 (catat di `reports/CHANGELOG.md`). Pengukuran biaya komputasi fine-tune penuh kemungkinan tidak muat; **tulis "tidak muat" apa adanya**, jangan memperkirakan.
   - Tanpa GPU NVIDIA: hanya E1, E2, E5, E6. Jangan menjalankan model Wav2Vec2-BERT di CPU tanpa persetujuan Faiz.
4. Jalankan `PYTHONPATH=src pytest -p no:cacheprovider`. Semua tes harus lulus (termasuk `test_frozen_backbone_trains_only_head` dan tes anti-leakage). Bila ada yang gagal, perbaiki atau laporkan sebelum training.
5. Jalankan setiap config dengan `--smoke` dulu. Run smoke tidak menulis ke `reports/results.csv`.

### 10.4 Ketahanan di laptop biasa
- Matikan sleep dan hibernasi selama run, colok charger, dan pantau suhu. Jalankan perintah panjang dengan log ke berkas (`nohup ... > logs/<nama>.log 2>&1 &` atau `tmux`), satu sifat atau satu eksperimen per sesi bila perlu.
- Run deep learning bisa dilanjutkan dengan `--resume <RUN_DIR>`. Jangan menghapus checkpoint epoch sebelum run selesai.
- Perbarui `reports/progress.md` (tanggal, eksperimen, status, run id, kendala) setiap selesai satu langkah, supaya sesi Claude Code baru dapat melanjutkan tanpa menebak.
- Waktu: proposal mencatat satu putaran penuh Wav2Vec2-BERT sekitar 8 hari di perangkat Faiz. Untuk E3 ukur dulu waktu satu epoch pada satu sifat, hitung perkiraan total (3 sifat × CV seed 42 + 3 seed uji + LIGO), **laporkan ke Faiz, dan tunggu persetujuan** sebelum menjalankan penuh.

### 10.5 Eksperimen, urut pengerjaan
**E1. XGBoost grid diperluas + 3 seed uji (CPU).** `python -m ta.train --config configs/xgb_mfcc_ext.yaml --task all` (setelah `--smoke`). Grid: n_estimators {300, 600, 1000} × max_depth {4, 6, 8} × learning_rate {0,03; 0,05; 0,1}; subsample dan colsample_bytree 0,8. Sel dipilih dari rata-rata macro-F1 CV seed 42, tanpa data uji. Uji akhir seed 42, 43, 44 dengan hiperparameter yang sama; keluaran `*_predictions_test_seeds.csv`. Periksa apakah sel terpilih masih di tepi grid baru; bila ya, catat sebagai keterbatasan dan **jangan** memperluas lagi tanpa bertanya.

**E2. Sensitivitas tanpa 26 berkas `-error` (CPU).** Ke-26 berkas itu (kolom `flag_error` di metadata, Data Benar folder 15) semuanya ada di data latih, bukan data uji, sehingga analisis pasca-hoc di data uji tidak berubah. Sensitivitas yang sebenarnya: latih ulang **XGBoost saja** dengan config terpisah `configs/xgb_mfcc_ext_noerror.yaml` (salinan E1 dengan opsi mengecualikan `flag_error`; tambahkan opsi itu seminimal mungkin dan beri tes), pada hiperparameter terpilih E1, lalu laporkan selisih macro-F1 uji terhadap E1. Jangan melatih ulang Wav2Vec2-BERT untuk ini.

**E3. Encoder Wav2Vec2-BERT beku + probe linear (GPU).** `python -m ta.train --config configs/w2v_bert_frozen.yaml --task all` (setelah `--smoke`). Checkpoint dan revisi sama dengan model utama; backbone beku dalam mode eval, hanya kepala linear yang dilatih (lr 1e-3, maks 10 epoch, patience 3; ditetapkan sebelum dijalankan, jangan diubah setelah melihat hasil). CV dan LIGO seed 42, uji akhir seed 42, 43, 44. Bila memori GPU habis, ikuti 10.3 butir 3.

**E4. Biaya komputasi (GPU dan CPU, satu mesin yang sama).** Buat `src/ta/cost.py` dan hasilnya ke `reports/cost/biaya_komputasi.csv` (satu baris per model dan metrik, plus kolom hardware dari `hardware_laptop.md`). Metrik (definisi di proposal subbab 3.5.1):
- Waktu latih per epoch (detik) untuk Wav2Vec2-BERT fine-tune penuh: ukur beberapa langkah optimizer pada konfigurasi model utama (batch efektif 32), ekstrapolasikan ke jumlah langkah per epoch data latih, dan tulis metodenya. Untuk XGBoost: waktu per putaran boosting pada hiperparameter terpilih E1. Bila fine-tune penuh tidak muat di VRAM, catat "tidak muat" dan jangan menebak.
- Jumlah parameter (total dan terlatih) untuk kedua model; untuk probe beku E3 laporkan juga jumlah parameter terlatih.
- Latensi inferensi per rekaman (milidetik, batch 1, **termasuk ekstraksi fitur**): 200 rekaman uji dipilih dengan seed tetap, 20 pemanasan, laporkan median dan persentil 95. XGBoost di CPU; Wav2Vec2-BERT di GPU, dan bila memungkinkan juga di CPU.
- Memori puncak GPU (MiB) saat satu langkah latih dan saat inferensi (`torch.cuda.max_memory_allocated`). Ini ukuran deskriptif tanpa uji statistik.

**E5. Analisis galat per huruf induk (CPU, bisa dikerjakan di komputer Faiz).** Proposal memakai 28 huruf: varian digabung ke huruf induk menurut `huruf_no` (2t→2, 5t→5, 8t→8, 16t→16, 21t→21, 26t→26, 25i/25ie/25ik→25). Label varian sudah dicek identik dengan induknya. Ubah `item_errors` di `src/ta/analysis.py` agar unit item adalah `huruf_no`, pindahkan keluaran lama ke `reports/analysis/archive_36item/` (jangan dikutip lagi), dan tulis ulang README analisis. Pastikan lipatan LIGO tetap 6 kelompok dengan varian ikut induknya.

**E6. Analisis statistik akhir (CPU).** Jalankan setelah E1 dan E3 selesai:
`PYTHONPATH=src python -m ta.analysis --xgb runs/20261102-004931_xgb_mfcc_all --w2v runs/20261109-100209_w2v_bert_all --frozen <RUN_BEKU> --xgb-seeds <RUN_XGB_EXT> --out reports/analysis`
(`--xgb-seeds` menambahkan hasil XGBoost 3 seed sebagai `xgb_3seed`; baca `--help` dan `reports/analysis/README.md`, dan pastikan Δ yang dilaporkan memakai rata-rata 3 seed untuk kedua model.) Keluaran: `metrik_uji_dan_acuan.csv`, `delta_ci_permutasi.csv`, `ligo_per_kategori.csv`, `galat_per_item.csv`. Angka lama diganti, bukan ditambah.

**E7 (opsional, hanya bila Faiz setuju). Baseline durasi saja.** Macro-F1 uji dari fitur durasi saja dan durasi + huruf + harakat dengan model sederhana yang dilatih di data latih saja, untuk menunjukkan bahwa model tidak sekadar belajar durasi. Laporkan terpisah dan jangan dimasukkan ke tabel utama.

### 10.6 Hal yang tidak boleh diubah
Split dan fold, seed uji 42/43/44, ambang keputusan 0,5, preprocessing (16 kHz, trim 40 dB, panjang tetap 1,9 detik), fitur MFCC (13 koefisien, FFT 512, 40 mel, 20 sampai 8.000 Hz, delta lebar 9, 78 ringkasan), checkpoint dan revisi Wav2Vec2-BERT, hiperparameter yang sudah ditetapkan di config, dan label (varian digabung ke induk hanya pada analisis per huruf).

### 10.7 Berhenti dan tanya Faiz bila
Jumlah data menyimpang dari 10.791 + 10.800 = 21.591 (21.586 setelah lima rekaman identik dikeluarkan); ada tes yang gagal; OOM yang tidak teratasi dengan `batch_size` dan `grad_accum`; sel grid terpilih di tepi grid baru; `bitsandbytes` tidak bisa dipasang; perkiraan waktu E3 lebih dari tiga hari; atau Anda terpikir mengubah sesuatu dari 10.6.

### 10.8 Metode statistik (tetap)
Unit analisis responden (4 Benar + 4 Salah di data uji). Bootstrap klaster bertingkat per kategori, B = 10.000, CI persentil 95%, dinyatakan terlalu sempit karena hanya 8 klaster. Uji permutasi tanda eksak pada selisih macro-F1 per responden (dua sisi, p terkecil 2/256). Koreksi Holm untuk tiga sifat. Δs = rata-rata macro-F1 Wav2Vec2-BERT (3 seed) dikurangi rata-rata macro-F1 XGBoost (3 seed), per sifat s. Hasil awal (XGBoost 1 seed) tidak boleh dikutip setelah E1 selesai.

### 10.9 Penyerahan hasil ke Faiz
1. Tulis `reports/RINGKASAN_LANJUTAN.md` (satu halaman): Δ macro-F1 per sifat (w2v-BERT vs XGBoost 3 seed, w2v-BERT vs encoder beku) dengan CI dan p Holm; LIGO per kategori; sensitivitas E2; biaya komputasi E4 beserta spesifikasi laptop; deviasi dari rencana; kendala. Tandai semuanya eksploratif. **Tulis juga daftar bagian proposal yang angkanya harus diganti Faiz** (subbab 3.5.1, 3.6, 3.1, jadwal), tanpa mengeditnya.
2. Tambahkan baris ke `reports/results.csv` untuk tiap run baru dan catat tiap run di `reports/CHANGELOG.md` (tanggal, config, commit, hasil, kendala). Semua angka berasal dari `runs/` atau `reports/`; jangan mengarang, membulatkan sendiri, atau memperkirakan. Eksperimen yang gagal atau tidak dijalankan ditulis apa adanya.
3. Kemas `hasil_eksperimen_lanjutan_<YYYYMMDD>.zip` berisi: run baru di `runs/` (**tanpa checkpoint model**), `reports/` (results.csv, CHANGELOG.md, progress.md, hardware_laptop.md, cost/, analysis/, RINGKASAN_LANJUTAN.md), dan config yang dipakai. Cetak ukuran dan SHA-256 berkasnya. Commit di branch `exp-lanjutan` dengan pesan bahasa Inggris; push hanya bila Faiz meminta.
4. Beri tahu Faiz jalur zip dan minta konfirmasi sebelum membersihkan laptop (10.1).
