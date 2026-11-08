# Data audit (TAHAP 2)

Dibuat otomatis oleh `python -m ta.data.audit`. Tidak ada rekaman yang dibuang oleh audit ini.

## 1. Jumlah berkas

- WAV di disk: 21600 ({'benar': 10800, 'salah': 10800})
- Baris metadata: 21591 ({'salah': 10800, 'benar': 10791})
- Di metadata tapi tidak ada di disk: 0
- Di disk tapi tidak ada di metadata: 9; sama dengan `rekaman_dikeluarkan.csv`: True
- Dikeluarkan karena kebocoran latih-uji (`exclude_leakage.csv`, keputusan user 2026-11-01): 5
- **Rekaman terpakai: 21586** (latih 17275, uji 4311)

## 2. Kualitas audio

Cache praproses: `data/interim/trim16k_35ecb48e72`, parameter: `{'sr': 16000, 'trim_top_db': 40.0, 'frame_ms': 25.0, 'hop_ms': 10.0, 'margin_ms': 50.0, 'peak': 0.95}`.

- Berkas gagal dibaca: 0
- Sample rate: {44100: 21583, 48000: 3}; cocok dengan metadata: True
- Durasi baca ulang vs metadata, selisih maks: 4.44e-16 s
- md5 sampel (float32) == kolom `md5` metadata: 21586 / 21586
- Berkas kosong (0 sampel): 0
- Hampir senyap (RMS < -60 dBFS): 8
- Ada sampel terpotong/clipping (|x| >= 0,999): 2 berkas; >1% sampel: 0
- RMS dBFS: median -34.6, p1 -44.6, p99 -25.4

| source_id | rms_dbfs_raw | dur_raw |
|---|---|---|
| salah/13/5t-2-1.wav | -61.443 | 0.114 |
| salah/13/5t-2-3.wav | -62.288 | 0.073 |
| salah/13/5t-2-4.wav | -61.842 | 0.084 |
| salah/13/5t-2-5.wav | -61.215 | 0.058 |
| salah/13/5t-3-1.wav | -62.376 | 0.108 |
| salah/13/5t-3-3.wav | -61.354 | 0.049 |
| salah/13/5t-3-4.wav | -62.233 | 0.075 |
| salah/13/5t-3-5.wav | -63.397 | 0.07 |

## 3. Durasi dan panjang tetap

| index | median | p1 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|
| mentah (semua) | 0.825 | 0.226 | 1.408 | 1.816 | 0.035 | 11.128 |
| setelah trim (semua) | 0.823 | 0.226 | 1.408 | 1.811 | 0.035 | 10.845 |
| setelah trim (latih) | 0.816 | 0.237 | 1.433 | 1.87 | 0.035 | 10.845 |

- Mentah > 2.0 s: 110; < 0.3 s: 754
- Setelah trim > 2.0 s: 109; < 0.3 s: 754
- **Persentil 99 durasi setelah trim (data latih) = 1.870 s -> panjang tetap = 1.9 s (30400 sampel)**, dibulatkan ke atas ke 0,1 s.
- Rekaman yang terpotong (tengah) pada panjang ini: 157 (0.73%); sisanya di-pad nol di kanan.

Outlier durasi mentah ditulis ke `reports/outlier_durasi.csv` (864 baris). **Belum ada yang dibuang**; menunggu keputusan user.

| jenis | category | split | n |
|---|---|---|---|
| panjang(>2s) | benar | latih | 20 |
| panjang(>2s) | salah | latih | 89 |
| panjang(>2s) | salah | uji | 1 |
| pendek(<0.3s) | benar | latih | 2 |
| pendek(<0.3s) | benar | uji | 6 |
| pendek(<0.3s) | salah | latih | 472 |
| pendek(<0.3s) | salah | uji | 274 |

Durasi setelah trim per kategori:

| category | count | 1% | 50% | 99% | max |
|---|---|---|---|---|---|
| benar | 10786.0 | 0.456 | 0.945 | 1.656 | 2.953 |
| salah | 10800.0 | 0.196 | 0.612 | 1.929 | 10.845 |

**Risiko jalan pintas:** rekaman sangat pendek dan sangat panjang didominasi Data Salah. Karena label ditentukan oleh (item, harakat, benar/salah), durasi bisa membantu model menebak status benar/salah tanpa belajar sifat artikulasi. Fitur MFCC (mean + std) dihitung hanya pada frame valid (tanpa padding), jadi durasi tidak masuk langsung sebagai fitur, tetapi skor per subset Data Benar/Data Salah tetap dilaporkan untuk memantau efek ini.

Sebaran outlier pendek per responden (10 terbanyak):

| responden | n |
|---|---|
| salah-12 | 180 |
| salah-15 | 169 |
| salah-10 | 140 |
| salah-11 | 134 |
| salah-13 | 91 |
| salah-16 | 24 |
| salah-14 | 6 |
| benar-9 | 4 |
| benar-11 | 2 |
| benar-13 | 2 |

## 4. Pemeriksaan kebocoran isi (md5)

- md5 sampel: unik 20217 dari 21586; sama latih-uji: **0**; lintas fold: 18
- md5 byte berkas: unik 20217 dari 21586; sama latih-uji: **0**; lintas fold: 18

md5 lintas fold di data latih (dibiarkan, dicatat sebagai keterbatasan CV):

| grup_cv | n_md5 |
|---|---|
| salah-13_15 <-> salah-16 | 9 |
| salah-17 <-> salah-18 | 6 |
| benar-6 <-> benar-7 | 3 |

Duplikat isi di dalam satu `grup_cv` (pengulangan identik) tidak memengaruhi split; tidak ada duplikat yang berbeda item, harakat, atau label.

## 5. Sebaran label dan split

| sifat | bagian | kelas0 | kelas1 | porsi1 |
|---|---|---|---|---|
| S1 | latih | 8635 | 8640 | 0.5001 |
| S1 | uji | 2154 | 2157 | 0.5003 |
| S1 | semua | 10789 | 10797 | 0.5002 |
| S3 | latih | 8640 | 8635 | 0.4999 |
| S3 | uji | 2151 | 2160 | 0.501 |
| S3 | semua | 10791 | 10795 | 0.5001 |
| T2 | latih | 8640 | 8635 | 0.4999 |
| T2 | uji | 2151 | 2160 | 0.501 |
| T2 | semua | 10791 | 10795 | 0.5001 |

Data latih seimbang (porsi kelas 1 sekitar 0,5) untuk ketiga sifat, jadi **class weight tidak dipakai**.

| split | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| latih | 0 | 3240 | 3780 | 3235 | 3780 | 3240 |
| uji | 4311 | 0 | 0 | 0 | 0 | 0 |

- `grup_cv` di lebih dari satu fold: 0; di lebih dari satu split: 0
- `responden`: 40 responden menjadi 37 `grup_cv` (digabung: benar-8_18_19, salah-13_15); responden di lebih dari satu fold: 0; di lebih dari satu split: 0; irisan responden latih dan uji: 0

Porsi kelas 1 per fold latih:

| fold | y_S1 | y_S3 | y_T2 |
|---|---|---|---|
| 1 | 0.5 | 0.5 | 0.5 |
| 2 | 0.532 | 0.536 | 0.532 |
| 3 | 0.501 | 0.499 | 0.499 |
| 4 | 0.468 | 0.464 | 0.468 |
| 5 | 0.5 | 0.5 | 0.5 |

## 6. Lipatan leave-item-group-out

Disimpan di `data/splits/ligo_items.csv` dan `data/splits/ligo_folds.csv`. Varian ikut huruf dasarnya. Evaluasi: latih = responden latih dengan item di luar lipatan k; uji = responden uji dengan item di lipatan k.

| ligo_fold | items | rekaman_latih_item_ini | rekaman_uji_item_ini | huruf S1=1 (benar) | huruf S3=1 (benar) | huruf T2=1 (benar) |
|---|---|---|---|---|---|---|
| 1 | 2, 2t, 3, 9, 15, 23 | 2880 | 720 | 1 | 1 | 1 |
| 2 | 4, 5, 5t, 11, 17, 24 | 2880 | 720 | 1 | 1 | 1 |
| 3 | 6, 7, 8, 8t, 10, 18 | 2880 | 720 | 2 | 2 | 1 |
| 4 | 1, 12, 16, 16t, 19, 28 | 2880 | 717 | 1 | 2 | 1 |
| 5 | 13, 14, 20, 21, 21t, 27 | 2875 | 717 | 4 | 2 | 1 |
| 6 | 22, 25i, 25ie, 25ik, 26, 26t | 2880 | 717 | 1 | 0 | 0 |

Lipatan 6 tidak berisi huruf qalqalah atau isti'la (Data Benar), karena hanya ada 5 huruf qalqalah untuk 6 lipatan. Kelas 1 di lipatan itu tetap ada lewat Data Salah (label dibalik).

## 7. Asumsi

- Responden `benar-N` dan `salah-N` diperlakukan sebagai identitas berbeda, mengikuti split bawaan dataset (belum dipastikan apakah orangnya sama).
- Arti akhiran varian (`i`, `ie`, `ik`, `t`) belum diketahui; varian diperlakukan sebagai item biasa.
- 26 berkas `*-error` di `Data Benar/15` diperlakukan sebagai pengucapan benar (sesuai dataset).
