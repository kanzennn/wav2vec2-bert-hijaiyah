# Analisis lanjutan (eksploratif)

Dihitung oleh `src/ta/analysis.py` dari berkas prediksi di `runs/` (tanpa audio). Semua hasil **eksploratif**: data uji
(8 responden) sudah pernah dilihat sebelum rencana ini ditulis (subbab 3.1 proposal). Unit independen = responden;
dengan 8 klaster, CI bootstrap cenderung terlalu sempit dan p permutasi terkecil = 2/256 = 0,0078.

Perintah:
`PYTHONPATH=src python -m ta.analysis --xgb runs/20261102-004931_xgb_mfcc_all --w2v runs/20261109-100209_w2v_bert_all --out reports/analysis`
(tambahkan `--frozen RUN` dan `--xgb-seeds RUN` setelah dua eksperimen pelatihan di bawah selesai).

| Berkas | Isi |
|---|---|
| `metrik_uji_dan_acuan.csv` | macro-F1, balanced accuracy, akurasi per model/sifat/kategori + aturan acuan "kategori saja" dan "item+harakat saja" |
| `delta_ci_permutasi.csv` | Δ macro-F1 (w2v-BERT − XGBoost), CI 95% bootstrap klaster (B=10.000), p permutasi tanda eksak, p Holm; varian sensitivitas |
| `ligo_per_kategori.csv` | LIGO per kategori (Benar/Salah/semua) dan per lipatan; AUC pooled dengan CI klaster |
| `galat_per_item.csv` | galat per item dengan CI bootstrap responden; kolom `ci_terpisah_dari_peringkat_6` menandai peringkat yang boleh diklaim |

## Hasil utama (data uji, w2v-BERT = rata-rata 3 seed, XGBoost = seed 42)

| Sifat | macro-F1 XGB | macro-F1 w2v-BERT | Δ | CI 95% | p permutasi | p Holm |
|---|---:|---:|---:|---|---:|---:|
| S1 | 0,6885 | 0,7886 | 0,1001 | [−0,048; 0,239] | 0,156 | 0,156 |
| S3 | 0,6678 | 0,7615 | 0,0937 | [0,044; 0,147] | 0,016 | 0,047 |
| T2 | 0,6620 | 0,7776 | 0,1157 | [0,005; 0,218] | 0,063 | 0,125 |

- Aturan acuan "kategori saja" (0,7223 / 0,7495 / 0,7216) lebih tinggi daripada XGBoost pada ketiga sifat dan hanya
  sedikit di bawah w2v-BERT (selisih 0,066 / 0,012 / 0,056); "item+harakat saja" ≈ 0,333. Jadi XGBoost belum mengalahkan
  detektor Benar/Salah sempurna, dan w2v-BERT unggul tipis di S3.
- Hanya S3 yang melewati Holm (0,047), dan itu tipis. CI S1 mencakup nol. Jangan menulis "signifikan" tanpa data
  konfirmatori (responden baru).
- LIGO subset **Data Salah**: AUC S3/T2 w2v-BERT 0,4715 / 0,4760 dan XGBoost 0,4899 / 0,5247, CI mencakup 0,5, jadi label
  Data Salah untuk item baru tidak bisa ditebak dari akustik. LIGO subset **Data Benar** (yang boleh ditafsirkan sebagai
  generalisasi sifat): AUC w2v-BERT S1/S3/T2 = 0,802 / 0,867 / 0,921, XGBoost 0,619 / 0,720 / 0,777.
- Sensitivitas tanpa berkas `-error`: **identik dengan hasil utama karena ke-26 berkas itu semuanya di data latih
  (benar-15), tidak ada di data uji.** Analisis ini baru bermakna bila model dilatih ulang tanpa berkas tersebut.
- Sensitivitas tanpa salinan identik di dalam folder (114 rekaman uji dibuang, n=4.197): Δ berubah ≤ 0,004; kesimpulan sama.
- Galat per item: tidak ada item yang CI-nya terpisah dari peringkat ke-6, jadi proposal tidak boleh mengklaim "lima item
  tersulit" sebagai peringkat pasti.
- Catatan: XGBoost baru satu seed; Δ membandingkan rata-rata 3 seed w2v-BERT dengan 1 run XGBoost sampai eksperimen 1
  di bawah dijalankan.

## Eksperimen pelatihan yang perlu dijalankan di laptop (butuh audio + GPU, tidak bisa dijalankan di sini)

1. **XGBoost grid diperluas + 3 seed uji** (menjawab "optimum di tepi grid" dan N-06). CPU saja, 27 sel grid × 5 lipatan × 3 sifat:
   `python -m ta.train --config configs/xgb_mfcc_ext.yaml --task all`
   Menghasilkan `*_predictions_test_seeds.csv` (3 seed) di folder run.
2. **Pembanding encoder beku + probe linear** (checkpoint sama, hanya kepala dilatih; menjawab J-05):
   `python -m ta.train --config configs/w2v_bert_frozen.yaml --task all`
   Opsi `freeze_backbone` baru di `src/ta/models/wav2vec_bert.py`; tes `test_frozen_backbone_trains_only_head` ditambahkan
   di `tests/test_dl.py` (belum dijalankan karena torch tidak ada di mesin peninjau; jalankan `pytest tests/test_dl.py`).
3. Setelah keduanya selesai, jalankan ulang `ta.analysis` dengan `--frozen` dan `--xgb-seeds` (otomatis menambah Δ
   w2v-BERT vs beku, beku vs XGBoost, dan w2v-BERT vs XGBoost 3 seed).
4. (Opsional, murah) Latih ulang XGBoost tanpa 26 berkas `-error` untuk sensitivitas yang sebenarnya.
