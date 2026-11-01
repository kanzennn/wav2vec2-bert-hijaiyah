# CHANGELOG

## 2026-11-01 — TAHAP 1 (ekstraksi dan inventarisasi)

- Dataset diekstrak dari `Hijaiyah Letter Pronunciation Audio Dataset.zip` ke `data/raw/` (21.600 WAV) dan `data/labels/`.
  Hasil verifikasi ada di `reports/label_inventory.md`.
- Tidak ditemukan hasil lama konsep identifikasi huruf 36 kelas. Folder `reports/archive_letter_id/` dan `runs/archive_letter_id/` hanya berisi README.
- Ditemukan 1 rekaman identik (md5 `ce66fdfe...`, item `21t`, kasrah) yang ada di latih (`benar-11`) dan uji (`benar-12`, `benar-13`).
  **Keputusan user:** 5 salinan di data latih dikeluarkan lewat `data/labels/exclude_leakage.csv`. Data uji tidak diubah.
  Total rekaman terpakai: 21.586 (latih 17.275, uji 4.311).
- 18 md5 lintas fold di dalam data latih dibiarkan dan dicatat sebagai keterbatasan validasi silang.
