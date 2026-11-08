# Arsip: model di luar cakupan CLAUDE.md versi 5

Diarsipkan 2026-11-09 setelah cakupan TA dikecilkan menjadi XGBoost (MFCC) vs Wav2Vec2-BERT (atas arahan dosen pembimbing).
Run di sini dibuat 2026-11-01 sampai 2026-11-02 dengan CLAUDE.md versi 4, split dan praproses yang sama (1,9 s, 21.586 rekaman).
Bukan bagian hasil utama; jangan dicampur dengan `reports/results.csv`.

| Run | Model | Catatan |
|---|---|---|
| `20261101-225111_svm_mfcc_all` | SVM RBF + MFCC | commit 0a70379 |
| `20261101-234341_rf_mfcc_all` | Random Forest + MFCC | commit e708014 |
| `20261102-011321_cnn_mel_all` | CNN + log-Mel, 3 seed | commit ea8df55 |
| `20261102-025730_crnn_mel_all` | CRNN + log-Mel, 3 seed | commit ea8df55 |
| `20261102-074714_w2v_frozen_all` | wav2vec2-base beku + kepala MLP, 3 seed | commit 0974578 |
| `20261102-083223_w2v_finetune_all` | wav2vec2-base fine-tune | **tidak selesai**: proses mati (exit code 4) setelah CV S1 seed 42 (macro-F1 CV 0,783 +- 0,105); tidak ada hasil uji/LIGO |
| `*_smoke` | uji asap | tidak di-commit |

Baris hasilnya ada di `reports/archive_out_of_scope/results.csv`. Kode modelnya dihapus dari `src/` pada commit pengarsipan ini dan tetap ada di riwayat git (commit 26d28ed).
