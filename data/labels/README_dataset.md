# Hijaiyah Letter Pronunciation Audio Dataset

Recordings of hijaiyah letters pronounced correctly ("Benar") and incorrectly ("Salah"), annotated with articulation characteristics (sifat huruf).

## Files
| File | Content |
|---|---|
| `Data_Benar.zip` | 10,800 WAV recordings, correct pronunciation, folders `Data Benar/1` ... `Data Benar/20` (one folder = one respondent, 540 files each) |
| `Data_Salah.zip` | 10,800 WAV recordings, incorrect pronunciation, folders `Data Salah/1` ... `Data Salah/20` |
| `metadata_rekaman.csv` | One row per recording used in the experiments (21,591): respondent, letter, vowel mark, repetition, sampling rate, duration, MD5, labels S1/S3/T2, train/test split, CV fold |
| `label_per_huruf.csv` | Reference labels of the 12 label variables (S1-S5, T1-T7) for the 28 letters |
| `rekaman_dikeluarkan.csv` | 9 recordings (Data Benar/12, `4-100.wav` ... `4-108.wav`) excluded from the experiments because the file names do not follow the naming scheme |
| `Kode_Pipeline.zip` | Python code: preprocessing, augmentation, feature extraction, splitting, models (paths in `config.py` must be adapted) |

## File naming
`<item>-<vowel>-<repetition>.wav`, e.g. `21-1-1.wav` = item 21 (letter qaf), vowel 1 (fathah), repetition 1.
Item = letter number 1-28 (optionally with a suffix for letter variants); vowel: 1 = fathah, 2 = kasrah, 3 = dhammah; repetition 1-5.
36 items x 3 vowels x 5 repetitions = 540 recordings per respondent.

## Labels
18 articulation characteristics (sifat huruf, Al-Hafiz, 2014) are encoded in 12 label variables; 5 for opposing pairs (S) and 7 for characteristics without an opposite (T):

| Variable | Coding |
|---|---|
| S1 | jahr = 0 / hams = 1 |
| S2 | rakhawah = 0 / bayniyyah = 1 / syiddah = 2 |
| S3 | istifal (tarqiq) = 0 / isti'la (tafkhim) = 1 |
| S4 | infitah = 0 / ithbaq = 1 |
| S5 | ishmat = 0 / idzlaq = 1 |
| T1 | shafir (1 = present, 0 = absent) |
| T2 | qalqalah |
| T3 | liin |
| T4 | inhirah |
| T5 | takrir |
| T6 | tafasysyi |
| T7 | istithaalah |

- `label_per_huruf.csv`: the reference label of all 12 variables for each of the 28 letters (correct pronunciation).
- `metadata_rekaman.csv`: per-recording labels `y_S1`, `y_S3`, `y_T2` (the three properties used in the accompanying article). S1 and T2 follow the letter. S3 additionally depends on the vowel for letters 7, 10 and 19 (tafkhim, except with kasrah), so it can differ from the letter-level S3 in `label_per_huruf.csv` for these letters.
- For the incorrect class, `y_S1`, `y_S3` and `y_T2` are the inverse of the correct label of the same item.

## Notes
- Audio: mono WAV, mostly 44.1 kHz (3 files at 48 kHz), about 0.7-1.0 s each.
- 26 files named `*-error` in `Data Benar/15` are treated as correct pronunciation.
- Several respondent folders contain identical recordings (e.g. Data Benar 8/18/19 and Data Salah 13/15); `metadata_rekaman.csv` column `grup_cv` merges them into one group. Split by respondent/`grup_cv`, never by recording, and let augmented copies follow their source recording.
- The augmented data are not included; they can be regenerated with the code.

## License
CC BY 4.0
