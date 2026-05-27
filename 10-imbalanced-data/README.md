# Modul 10 · Menangani Data Tidak Seimbang (Imbalanced Data)

> Hampir semua masalah ML bernilai tinggi punya kelas tidak seimbang: deteksi fraud (0.1% penipuan), churn, diagnosis penyakit langka, deteksi cacat produksi. Model standar akan "malas" — memprediksi kelas mayoritas saja sudah akurasi 99%, tapi **gagal total** untuk tujuan sebenarnya. Modul ini soal membuat model benar-benar belajar kelas minoritas.

## Tujuan Belajar
- Memahami kenapa imbalance merusak model & metrik.
- Memilih **metrik yang benar** untuk kasus tidak seimbang.
- Menguasai 3 strategi: **level data** (resampling), **level algoritma** (class weight), **level keputusan** (threshold).
- Menerapkan **SMOTE** & variannya dengan benar (tanpa leakage!).
- Tahu kapan memakai apa.

## Daftar Isi
1. Kenapa imbalance jadi masalah
2. Metrik yang benar (jangan pakai akurasi)
3. Strategi 1 — Level Data (resampling & SMOTE)
4. Strategi 2 — Level Algoritma (class weight)
5. Strategi 3 — Level Keputusan (threshold tuning)
6. Jebakan: leakage saat resampling
7. Anomaly detection sebagai alternatif
8. Alur kerja & cheat sheet

---

## 1. Kenapa Imbalance Jadi Masalah

Misal 990 transaksi normal, 10 fraud (99% : 1%).

- **Loss function** mengoptimalkan rata-rata → model "rugi sedikit" kalau mengorbankan 10 fraud demi 990 normal.
- Model "selalu prediksi normal" → **akurasi 99%**, tapi **recall fraud = 0%**. Tidak berguna.
- Kelas minoritas sering justru yang **paling penting & paling mahal** kalau salah (fraud lolos, kanker tak terdeteksi).

> Inti masalah: akurasi tinggi menyembunyikan kegagalan total pada kelas yang kita pedulikan.

---

## 2. Metrik yang Benar (JANGAN pakai akurasi)

Lihat juga [Modul 01](../01-fondasi-ml/README.md) & [Modul 09](../09-evaluasi-interpretasi/README.md).

| Metrik | Kenapa cocok untuk imbalanced |
|---|---|
| **Recall** (kelas minoritas) | berapa fraud yang berhasil ditangkap (FN mahal) |
| **Precision** | dari yang ditandai fraud, berapa yang benar (FP mahal) |
| **F1 / F-beta** | keseimbangan; F2 menekankan recall lebih dari precision |
| **PR-AUC** (Precision-Recall AUC) | **lebih informatif dari ROC-AUC** saat sangat tidak seimbang |
| **Confusion matrix** | selalu lihat angka mentahnya |

```python
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))           # fokus recall/precision kelas 1
print("PR-AUC:", average_precision_score(y_test, y_proba))   # y_proba kelas positif
```

> ⚠️ **ROC-AUC bisa terlihat bagus (mis. 0.95) padahal model buruk** untuk minoritas, karena ROC didominasi True Negative. Untuk imbalance ekstrem, **PR-AUC** lebih jujur.

---

## 3. Strategi 1 — Level Data (Resampling)

Ubah komposisi data training agar lebih seimbang. Pakai library `imbalanced-learn` (imblearn).

### a) Random Undersampling — buang sebagian kelas mayoritas
Cepat, tapi **membuang informasi**. Cocok kalau data sangat banyak.

### b) Random Oversampling — duplikasi kelas minoritas
Sederhana, tapi rawan **overfitting** (model menghafal duplikat).

### c) SMOTE (Synthetic Minority Over-sampling) ⭐
Tidak menduplikasi, tapi **membuat sampel sintetis baru** dengan interpolasi antar tetangga minoritas. Jauh lebih baik dari oversampling biasa.

```python
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from collections import Counter

print("Sebelum:", Counter(y_train))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)   # HANYA di data train!
print("Sesudah:", Counter(y_res))
```

**Varian SMOTE:**
- **BorderlineSMOTE** — fokus pada sampel minoritas dekat batas keputusan (yang sulit).
- **ADASYN** — buat lebih banyak sintetis di area yang sulit dipelajari.
- **SMOTENC** — untuk dataset dengan **fitur kategorikal** (SMOTE biasa hanya numerik).
- **SMOTE + Tomek / SMOTEENN** — gabung oversampling minoritas + bersihkan noise mayoritas.

```python
from imblearn.combine import SMOTETomek
X_res, y_res = SMOTETomek(random_state=42).fit_resample(X_train, y_train)
```

> 📌 **SMOTE bekerja dengan jarak** → wajib pada fitur **numerik & ter-scale**. Untuk fitur kategorikal pakai SMOTENC.

---

## 4. Strategi 2 — Level Algoritma (Class Weight)

Daripada mengubah data, **beri bobot lebih besar** pada kesalahan di kelas minoritas dalam loss function. Sering jadi pilihan **pertama** karena sederhana & tanpa risiko leakage resampling.

```python
# scikit-learn: tinggal set class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

LogisticRegression(class_weight="balanced")           # otomatis ~ inversi frekuensi
RandomForestClassifier(class_weight="balanced")
# atau manual: class_weight={0: 1, 1: 50}

# XGBoost / LightGBM:
from xgboost import XGBClassifier
n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
XGBClassifier(scale_pos_weight=n_neg / n_pos)         # rasio kelas

# PyTorch (deep learning):
import torch.nn as nn
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 50.0]))
```

> `class_weight="balanced"` memberi bobot ~ `n_total / (n_kelas × n_sampel_kelas)` — kelas langka otomatis lebih "mahal" untuk disalahkan.

---

## 5. Strategi 3 — Level Keputusan (Threshold Tuning)

Sering diabaikan tapi **sangat ampuh & gratis**. Default klasifikasi pakai threshold 0.5. Untuk imbalance, threshold itu hampir selalu salah.

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

proba = model.predict_proba(X_val)[:, 1]
prec, rec, thresholds = precision_recall_curve(y_val, proba)

# Pilih threshold yang memaksimalkan F1 (atau sesuai trade-off bisnis):
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_t = thresholds[np.argmax(f1)]
print("Threshold optimal:", best_t)

y_pred = (model.predict_proba(X_test)[:, 1] >= best_t).astype(int)
```

> **Keputusan bisnis, bukan teknis:** kalau melewatkan fraud sangat mahal, turunkan threshold (tangkap lebih banyak, terima lebih banyak false alarm = recall ↑ precision ↓). Plot **PR-curve** dan pilih titik sesuai biaya FP vs FN.

---

## 6. Jebakan Kritis — Leakage saat Resampling

🚨 **Kesalahan #1 pemula:** melakukan SMOTE pada **seluruh data sebelum split** atau di dalam cross-validation dengan cara salah. Sampel sintetis bocor ke validation → skor terlihat bagus tapi **bohong**.

**Aturan:** resampling **HANYA** pada fold training, **tidak pernah** pada validation/test. Pakai `Pipeline` dari imblearn agar otomatis benar:

```python
from imblearn.pipeline import Pipeline as ImbPipeline   # PENTING: dari imblearn, bukan sklearn
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),     # hanya dijalankan pada train tiap fold
    ("clf", RandomForestClassifier(random_state=42)),
])

# SMOTE otomatis hanya diterapkan ke bagian train di tiap fold -> tanpa leakage
scores = cross_val_score(pipe, X, y, cv=5, scoring="average_precision")
print(f"PR-AUC: {scores.mean():.3f} ± {scores.std():.3f}")
```

> Test set **tidak pernah** di-resample — ia harus mencerminkan distribusi dunia nyata yang tetap timpang.

---

## 7. Anomaly Detection sebagai Alternatif

Saat imbalance **ekstrem** (mis. <0.1%), kadang lebih baik perlakukan sebagai **deteksi anomali** daripada klasifikasi: model hanya belajar "normal", lalu menandai apa pun yang menyimpang.

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.01, random_state=42).fit(X_train_normal)
anomali = iso.predict(X_test) == -1     # -1 = anomali

# Alternatif: One-Class SVM, Local Outlier Factor, Autoencoder (deep learning)
```
Cocok untuk fraud/intrusion/cacat di mana contoh "positif" sangat sedikit atau tak terduga bentuknya.

---

## 8. Alur Kerja & Cheat Sheet

```
1. JANGAN pakai akurasi → pilih recall/precision/F1/PR-AUC sesuai biaya bisnis.
2. Mulai paling sederhana: class_weight="balanced"  (tanpa risiko leakage).
3. Tuning THRESHOLD pakai PR-curve — sering ini sudah cukup & gratis.
4. Kalau masih kurang: resampling via imblearn Pipeline (SMOTE/SMOTENC/SMOTETomek).
5. Imbalance ekstrem (<0.1%) → pertimbangkan anomaly detection.
6. Evaluasi SELALU di test set asli yang tetap timpang.
```

| Situasi | Saran utama |
|---|---|
| Imbalance ringan (mis. 70:30) | class_weight + threshold tuning |
| Imbalance sedang (95:5) | class_weight + SMOTE + threshold |
| Fitur kategorikal | SMOTENC |
| Imbalance ekstrem (<0.5%) | anomaly detection / class_weight kuat |
| Data sangat banyak | undersampling mayoritas bisa cukup |

> 💡 **Jangan kombinasi membabi buta.** Coba satu strategi, ukur dengan metrik yang benar, baru tambah kompleksitas. Sering `class_weight` + threshold tuning sudah mengalahkan SMOTE yang rumit.

## Latihan
1. Ambil dataset fraud (mis. Credit Card Fraud di Kaggle) atau buat imbalance sintetis dengan `make_classification(weights=[0.99, 0.01])`. Tunjukkan model default punya akurasi tinggi tapi recall ~0.
2. Bandingkan 4 pendekatan: baseline, class_weight, SMOTE (via Pipeline), threshold tuning. Buat tabel recall/precision/PR-AUC.
3. Tunjukkan bahaya leakage: lakukan SMOTE sebelum split vs di dalam Pipeline. Bandingkan skor CV — yang mana yang menipu?
4. Plot PR-curve, pilih threshold untuk skenario "FN 10× lebih mahal dari FP". Tafsirkan hasilnya.
5. Coba IsolationForest pada kasus imbalance ekstrem. Bandingkan dengan klasifikasi berbobot.

⬅️ Kembali ke [Daftar Modul](../README.md) · Terkait: [Modul 02 · Data](../02-data-feature-engineering/README.md), [Modul 09 · Evaluasi](../09-evaluasi-interpretasi/README.md)
