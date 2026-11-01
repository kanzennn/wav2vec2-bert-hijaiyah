# Inventarisasi label dan data (TAHAP 1)

Tanggal: 2026-11-01. Semua angka dihitung langsung dari `data/labels/*.csv` dan berkas WAV di `data/raw/`
(skrip verifikasi read-only, belum masuk `src/`). Acuan: CLAUDE.md bagian 2.

## 1. Ekstraksi dan tata letak

| Langkah | Hasil |
|---|---|
| Zip sumber | `./Hijaiyah Letter Pronunciation Audio Dataset.zip` (1,08 GB), satu folder pembungkus `Hijaiyah Letter Pronunciation Audio Dataset for Ar/` |
| Isi zip | `Data_Benar.zip`, `Data_Salah.zip`, `metadata_rekaman.csv`, `label_per_huruf.csv`, `rekaman_dikeluarkan.csv`, `Kode_Pipeline.zip`, `README.md`, sesuai CLAUDE.md |
| Path berbahaya (`/`, `..`, huruf drive) | tidak ada, baik di zip luar maupun zip bersarang |
| `data/raw/Data Benar/1..20` | 10.800 WAV (20 folder x 540) |
| `data/raw/Data Salah/1..20` | 10.800 WAV (20 folder x 540) |
| `data/labels/` | `metadata_rekaman.csv`, `label_per_huruf.csv`, `rekaman_dikeluarkan.csv`, `README_dataset.md` (salinan README dataset) |
| `data/raw_extract/Kode_Pipeline/` | hanya dibaca sebagai acuan, tidak dijalankan |

Catatan: audio di `data/raw/` merupakan **salinan** dari `data/raw_extract/audio/`, karena perintah pemindahan (`mv`) diblokir
oleh izin. `data/raw_extract/` (sekitar 2 GB) masih ada dan bisa dihapus setelah user setuju.

## 2. Disk vs metadata

| Cek | Hasil | Sesuai CLAUDE.md |
|---|---|---|
| Baris metadata | 21.591 (benar 10.791, salah 10.800) | ya |
| Kolom | 20 kolom, nama dan urutan sama dengan bagian 2.2 | ya |
| `source_id` unik | ya | - |
| Ada di metadata tapi tidak ada di disk | 0 | ya |
| Ada di disk tapi tidak ada di metadata | 9: `benar/12/4-100.wav` ... `4-108.wav`, sama persis dengan `rekaman_dikeluarkan.csv` | ya |
| Sample rate | 44.100 Hz: 21.588; 48.000 Hz: 3 | ya |
| Format | semua int16 PCM mono | ya |
| `duration` metadata vs hasil baca ulang | selisih maks 9e-16 s | ya |
| Durasi | median 0,825 s; p1 0,226; p99 1,816; min 0,035; maks 11,128; >2 s: 110; <0,3 s: 754 | ya |
| `flag_error=True` | 26, semuanya di `benar/15` | ya |

**Catatan parsing nama berkas:** regex `^(\d+)([A-Za-z]*)-(\d)-(\d+)\.wav$` dari CLAUDE.md **tidak cocok** untuk 26 berkas
`*-error.wav` (mis. `11-3-4-error.wav`, `21t-3-5-error.wav`). Kolom `item`, `huruf_no`, `sub`, dan `rep` di metadata sudah
terisi benar untuk berkas-berkas itu. Rencana: indeks membaca kolom metadata, sedangkan regex hanya dipakai untuk cek silang
dengan akhiran opsional `(-error)?`.

## 3. Item, harakat, pengulangan

- 36 item: `1, 2, 2t, 3, 4, 5, 5t, 6, 7, 8, 8t, 9, 10, 11, 12, 13, 14, 15, 16, 16t, 17, 18, 19, 20, 21, 21t, 22, 23, 24, 25i, 25ie, 25ik, 26, 26t, 27, 28`. Tidak ada item `25` polos. **Sesuai.**
- `huruf_no` = prefiks angka `item` untuk semua baris (1-28). **Sesuai.**
- Per item per kategori 300 rekaman (100 per harakat), kecuali Data Benar `26t`, `27`, `28` = 297 (99 per harakat). **Sesuai.**
- Per responden 540 rekaman, kecuali `benar-12` = 531 (9 berkas dikeluarkan, semuanya `rep=4`).
- Harakat (`sub`) 1/2/3: masing-masing 7.197.
- Arti akhiran `i`, `ie`, `ik`, `t` **tidak dijelaskan** di README dataset maupun di kode acuan. Tidak ditebak; varian diperlakukan sebagai item biasa.

## 4. Sebaran label

| Sifat | Data Benar 0 / 1 | Gabungan 0 / 1 | Latih (fold 1-5) 0 / 1 | Uji (fold 0) 0 / 1 |
|---|---|---|---|---|
| S1 (jahr / hams) | 7.794 / 2.997 | 10.794 / 10.797 | 8.640 / 8.640 | 2.154 / 2.157 |
| S3 (istifal / isti'la) | 8.091 / 2.700 | 10.791 / 10.800 | 8.640 / 8.640 | 2.151 / 2.160 |
| T2 (tanpa / qalqalah) | 7.791 / 3.000 | 10.791 / 10.800 | 8.640 / 8.640 | 2.151 / 2.160 |

Semua angka **sama dengan tabel CLAUDE.md 2.2**. Data latih tepat 50:50 untuk ketiga sifat, jadi **class weight tidak diperlukan**.

Konsistensi label:
- Label konstan dalam setiap (kategori, item, harakat): ya, untuk ketiga sifat.
- Data Salah = 1 - Data Benar pada (item, harakat) yang sama: ya, untuk ketiga sifat (berarti label sudah dibalik dataset; jangan dibalik lagi).
- Data Benar vs `label_per_huruf.csv` (item polos 1-28): S1 dan T2 cocok semua; S3 cocok kecuali huruf 7, 10, 19. Untuk ketiganya, metadata = 1 pada fathah/dhammah dan 0 pada kasrah (CSV: خ=1, ر=0, غ=1). **Sesuai bagian 2.2.**
- Jumlah huruf kelas 1 di CSV: S1 hams 10 (ت ث ح خ س ش ص ف ك ه), S3 isti'la 7 (خ ص ض ط ظ غ ق), T2 qalqalah 5 (ب ج د ط ق). **Sesuai bagian 2.3.**
- Label varian (Data Benar): `25i`, `25ie`, `25ik`, `26t` = 0/0/0; `2t`, `5t`, `8t` = S1 0, S3 0, T2 1; `16t`, `21t` = S1 0, S3 1, T2 1. **Sesuai bagian 2.2.**

## 5. Split dan fold

| fold | 0 (uji) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| rekaman | 4.311 | 3.240 | 3.780 | 3.240 | 3.780 | 3.240 |
| `grup_cv` | 8 | 6 | 6 | 6 | 5 | 6 |

- `split=uji`: 4.311 rekaman = benar-2 (B1), benar-9 (B2), benar-12 (B3), benar-13 (B4), salah-2 (K1), salah-4 (K2), salah-10 (K3), salah-11 (K4). `kode_uji` kosong di luar split uji. **Sesuai.**
- `split=latih`: 17.280 rekaman, fold 1-5 berisi 6, 6, 6, 5, 6 `grup_cv`. **Sesuai.**
- Tidak ada `grup_cv` yang terpecah lintas split atau fold. Grup gabungan: `benar-8_18_19` (fold 4) dan `salah-13_15` (fold 2).

Isi fold:
- fold 1: benar-4, 5, 16; salah-8, 12, 16
- fold 2: benar-1, 6, 14; salah-1, 6, 13_15
- fold 3: benar-7, 11, 17; salah-9, 14, 19
- fold 4: benar-10, 8_18_19; salah-5, 18, 20
- fold 5: benar-3, 15, 20; salah-3, 7, 17

## 6. Duplikat isi (md5) — PERLU KEPUTUSAN

Kolom `md5` di metadata adalah md5 dari **sampel audio float32** (`soundfile.read(dtype="float32")`), **bukan** md5 byte
berkas. Hash sampel yang dihitung ulang cocok 21.591/21.591. Hash byte berkas menghasilkan struktur duplikat yang identik.

- md5 unik: 20.217 dari 21.591. Ada 1.176 kelompok duplikat (2.550 baris).
- Tidak satu pun kelompok duplikat berisi item, harakat, kategori, atau label yang berbeda. Semuanya hanya beda `rep` atau beda folder, jadi **tidak ada konflik label**.
- 1.157 kelompok berada di dalam satu `grup_cv` (pengulangan yang identik). Terbanyak di `benar-8_18_19` (1.280 baris) dan `salah-13_15` (120 baris). Ini aman untuk split.
- **19 md5 muncul di lebih dari satu fold:**
  - **1 md5 lintas latih ↔ uji (kebocoran):** `21t`, kasrah (sub 3), md5 `ce66fdfe...`, durasi 1,643 s. Muncul di `benar-11/21t-3-{1..5}` (latih, fold 3; 5 baris) dan di `benar-12/21t-3-{1..5}` serta `benar-13/21t-3-{3,4}` (uji; 7 baris).
  - 18 md5 lintas fold di dalam data latih (hanya memengaruhi CV): `salah-16` (fold 1) ↔ `salah-13_15` (fold 2), 9 rekaman `2t`/`21t`; `salah-17` (fold 5) ↔ `salah-18` (fold 4), 6 rekaman; `benar-6` (fold 2) ↔ `benar-7` (fold 3), 3 rekaman `21t` sub 3.

Ini menyimpang dari aturan 4.2 CLAUDE.md ("tidak ada isi sama di data latih dan uji").

**Keputusan user (2026-11-01):** 5 salinan di data latih (`benar/11/21t-3-1.wav` ... `21t-3-5.wav`, fold 3) dikeluarkan dari
eksperimen. Daftarnya disimpan di `data/labels/exclude_leakage.csv`. Berkas WAV dan CSV bawaan dataset tidak diubah;
pengecualian diterapkan saat indeks dibangun. Data uji tetap utuh. 18 md5 yang lintas fold di dalam data latih dibiarkan dan
dicatat sebagai keterbatasan CV.

Setelah pengecualian:

| | Total | Latih | Uji | fold 3 | `benar-11` |
|---|---|---|---|---|---|
| Rekaman | 21.586 | 17.275 | 4.311 | 3.235 | 535 |

| Sifat | Latih 0 / 1 | Gabungan 0 / 1 |
|---|---|---|
| S1 | 8.635 / 8.640 | 10.789 / 10.797 |
| S3 | 8.640 / 8.635 | 10.791 / 10.795 |
| T2 | 8.640 / 8.635 | 10.791 / 10.795 |

md5 latih yang sama dengan uji: 0. Data latih tetap seimbang (selisih 5), jadi class weight tetap tidak diperlukan. Jumlah
acuan di audit TAHAP 2 menjadi 21.591 baris metadata dikurangi 5 = 21.586 rekaman terpakai.

## 7. Asumsi yang dicatat

- Responden `benar-N` dan `salah-N` diperlakukan sebagai identitas berbeda, mengikuti split bawaan dataset (belum dipastikan apakah orangnya sama).
- Arti akhiran varian belum diketahui (menunggu konfirmasi Pak Aaz).
- `S2, S4, S5, T1, T3-T7` hanya ada per huruf (Data Benar). Konvensi Data Salah-nya tidak tersedia, jadi di luar tahap utama.

## 8. Environment (untuk TAHAP 2)

Python 3.11.0, numpy 1.26.4, pandas 3.0.3, scipy 1.17.1, scikit-learn 1.9.0, torch/torchaudio 2.6.0+cu124, PyYAML,
matplotlib; GPU RTX 4060 Laptop 8 GB. **Belum terpasang:** `soundfile`, `librosa`, `xgboost`, `pytest`, `transformers`.
