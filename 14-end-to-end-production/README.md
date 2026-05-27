# Modul 14 · Proyek Data Science End-to-End (Production-Grade)

> Tutorial "101" berhenti di `model.fit()` lalu `accuracy_score()`. Modul ini menunjukkan alur kerja yang **benar-benar dipakai di produksi**: dari membaca data sampai model yang siap di-deploy, dengan disiplin yang membuat hasilmu **reproducible, bebas leakage, terdokumentasi, dan bisa dipercaya**. Ini capstone yang menyatukan Modul 01–13.

## Tujuan Belajar
- Menjalankan proyek ML end-to-end dengan **standar profesional**.
- Menghindari kesalahan yang membuat model "bagus di notebook, gagal di produksi".
- Memahami **kenapa** tiap langkah dilakukan dengan urutan tertentu.

## Daftar Isi
1. Prinsip yang membedakan production dari tutorial
2. Struktur proyek & reproducibility
3. Langkah 0 — Framing masalah & metrik
4. Langkah 1 — Ingest data (data contract)
5. Langkah 2 — Split DULU (holdout discipline)
6. Langkah 3 — EDA (hanya di train)
7. Langkah 4 — Baseline dulu
8. Langkah 5 — Preprocessing dalam Pipeline (anti-leakage)
9. Langkah 6 — Validasi & tuning yang benar
10. Langkah 7 — Evaluasi mendalam di holdout
11. Langkah 8 — Interpretasi & model card
12. Langkah 9 — Persist pipeline + metadata
13. Langkah 10 — Serah terima ke serving
14. Checklist "production-approved" & jebakan

> 📂 Implementasi lengkap yang bisa dijalankan: [pipeline.py](./pipeline.py) — satu skrip end-to-end mengikuti semua prinsip di bawah, menghasilkan model + metadata + laporan.

---

## 1. Prinsip yang Membedakan Production dari Tutorial

| Tutorial 101 | Production-grade |
|---|---|
| Semua di satu notebook berantakan | Kode modular, fungsi, config terpisah |
| `train_test_split` lalu langsung fit | **Holdout dikunci**, validasi terstruktur |
| Scaling/encoding manual sebelum split | Semua transform di dalam **Pipeline** (anti-leakage) |
| Metrik = accuracy | Metrik **selaras tujuan bisnis** + analisis per-slice |
| "Akurasi 92%, selesai" | Error analysis, calibration, interpretasi, dokumentasi |
| Tidak reproducible | Seed tetap, versi tercatat, artifact + metadata |
| Model = `.pkl` tanpa konteks | Pipeline + metadata + model card + monitoring plan |

> **Mantra:** "Model bukan hasil akhir. Yang kamu kirim ke produksi adalah **pipeline + bukti bahwa ia bisa dipercaya**."

---

## 2. Struktur Proyek & Reproducibility

```
proyek-churn/
├── config.yaml              # semua parameter & path (BUKAN hardcode di kode)
├── data/
│   ├── raw/                 # data mentah — read-only, jangan pernah diubah
│   └── processed/
├── src/
│   ├── data.py              # ingest & validasi
│   ├── features.py          # feature engineering
│   ├── train.py             # training pipeline
│   └── evaluate.py          # evaluasi
├── models/                  # artifact tersimpan (+ metadata)
├── reports/                 # metrik, plot, model card
├── tests/                   # unit test untuk transform & data
├── requirements.txt         # versi terkunci (pip freeze)
└── README.md
```

**Reproducibility — non-negotiable:**
```python
import numpy as np, random, os
SEED = 42
random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)
# torch.manual_seed(SEED) untuk deep learning
```
Tanpa seed tetap + versi library tercatat, hasilmu **tak bisa diverifikasi orang lain** — itu diskualifikasi di produksi.

---

## 3. Langkah 0 — Framing Masalah & Metrik

Sebelum satu baris kode model:
- **Apa keputusan bisnis** yang model dukung? (mis. "siapa pelanggan yang akan churn agar tim retensi bisa intervensi")
- **Metrik bisnis** vs **metrik ML**. Churn: biaya FN (pelanggan hilang) vs FP (diskon sia-sia) → ini menentukan apakah optimasi **recall** atau **precision** (lihat [Modul 01](../01-fondasi-ml/README.md) & [Modul 10](../10-imbalanced-data/README.md)).
- **Definisi label** yang tegas & dapat diukur (churn = tidak transaksi 90 hari? definisikan!).
- **Batasan**: latency inferensi, interpretability (regulasi?), volume data.

> Salah framing di sini = seluruh proyek salah arah. Investasikan waktu di langkah ini.

---

## 4. Langkah 1 — Ingest Data (Data Contract)

Ambil data (sering via SQL, [Modul 12](../12-sql-untuk-ml/README.md)) lalu **validasi sebelum dipakai** — jangan percaya data mentah.

```python
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # DATA CONTRACT: validasi asumsi secara eksplisit
    assert df.shape[0] > 0, "Data kosong"
    assert {"customer_id", "target"}.issubset(df.columns), "Kolom wajib hilang"
    assert df["target"].isin([0, 1]).all(), "Target harus biner"
    assert df["customer_id"].is_unique, "customer_id duplikat!"
    return df
```
Library produksi: **Pandera** atau **Great Expectations** untuk validasi skema/distribusi otomatis. Validasi data gagal → **gagalkan pipeline lebih awal**, jangan diam-diam menghasilkan model buruk.

> ⚠️ **Buang identifier & kolom yang tak akan tersedia saat inferensi** (mis. `customer_id`, atau fitur yang baru ada *setelah* target diketahui — itu **leakage**).

---

## 5. Langkah 2 — Split DULU (Holdout Discipline)

🚨 **Aturan #1 anti-leakage:** pisahkan **test set sebelum melihat apa pun**. Test set adalah "masa depan" — disentuh **sekali** di akhir.

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target", "customer_id"])
y = df["target"]

# Test set dikunci. JANGAN EDA/fit/scaling di sini.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y    # stratify jaga proporsi kelas
)
```
- **Time series?** Split **kronologis**, bukan acak ([Modul 11](../11-time-series/README.md)).
- **Ada grup** (mis. banyak baris per pelanggan)? Pakai `GroupShuffleSplit` agar pelanggan yang sama tak bocor ke train & test sekaligus.

> Semua langkah berikut (EDA, preprocessing, tuning) **hanya** menyentuh `X_trainval`.

---

## 6. Langkah 3 — EDA (Hanya di Train)

Pahami data ([Modul 02](../02-data-feature-engineering/README.md)), tapi **hanya pada data train** — agar wawasan tak "bocor" dari test.

- Distribusi target (seimbang? → [Modul 10](../10-imbalanced-data/README.md)).
- Missing & outlier per kolom.
- Korelasi/asosiasi dengan target (sesuai tipe data — Pearson/Cramér's V/correlation ratio).
- **Cari kebocoran tersembunyi:** fitur yang korelasinya "terlalu sempurna" dengan target sering = leakage.

```python
X_trainval.describe(include="all")
X_trainval.isnull().mean().sort_values(ascending=False)
y_trainval.value_counts(normalize=True)   # cek imbalance
```

---

## 7. Langkah 4 — Baseline Dulu

**Selalu** mulai dari model paling sederhana. Tanpa baseline, kamu tak tahu apakah model rumitmu benar-benar menambah nilai.

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# Baseline naif: selalu tebak kelas mayoritas
dummy = DummyClassifier(strategy="most_frequent")
# Baseline waras: regresi logistik sederhana
baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
```
Aturan: model finalmu **harus mengalahkan** baseline secara meyakinkan pada metrik bisnis — kalau tidak, pakai baseline (lebih murah & sederhana).

---

## 8. Langkah 5 — Preprocessing dalam Pipeline (Anti-Leakage)

🚨 **Aturan #2 anti-leakage:** semua transform yang "belajar" dari data (imputasi, scaling, encoding, SMOTE) **harus di dalam Pipeline** agar di-`fit` hanya pada fold train, tidak pernah pada validation/test.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
```
> Kalau pakai resampling (SMOTE), gunakan `imblearn.pipeline.Pipeline` agar resampling **hanya** di fold train ([Modul 10](../10-imbalanced-data/README.md)). `handle_unknown="ignore"` mencegah crash saat kategori baru muncul di produksi.

---

## 9. Langkah 6 — Validasi & Tuning yang Benar

Pilih model & hyperparameter via **cross-validation pada train**, dengan metrik yang benar.

```python
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

param_dist = {
    "clf__C": [0.01, 0.1, 1, 10],
    # untuk model lain: clf__n_estimators, clf__max_depth, dst.
}
search = RandomizedSearchCV(
    model, param_dist, n_iter=10, cv=cv,
    scoring="average_precision",      # PR-AUC: cocok untuk imbalance
    random_state=SEED, n_jobs=-1, refit=True,
)
search.fit(X_trainval, y_trainval)    # preprocessing otomatis fit per-fold -> aman
best_model = search.best_estimator_
```

**Yang sering salah:**
- Tuning pakai test set → **leakage**. Tuning **hanya** di train (lewat CV).
- Metrik salah untuk imbalance (accuracy) → pakai PR-AUC/F1/recall ([Modul 09](../09-evaluasi-interpretasi/README.md), [10](../10-imbalanced-data/README.md)).
- Untuk estimasi performa **tak bias** saat tuning, idealnya **nested CV** (outer untuk estimasi, inner untuk tuning).

---

## 10. Langkah 7 — Evaluasi Mendalam di Holdout

Baru **sekarang** sentuh test set — **sekali**. Jangan kembali utak-atik model setelah ini (kalau iya, test set kotor).

```python
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (proba >= threshold).astype(int)     # threshold dipilih dari train, bkn test!

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("PR-AUC:", average_precision_score(y_test, proba))
```
Lakukan evaluasi lengkap ala [Modul 09](../09-evaluasi-interpretasi/README.md):
- Confusion matrix + metrik per-kelas, **bandingkan vs baseline**.
- **Threshold** dipilih dari validation (biaya bisnis FP vs FN), bukan dari test.
- **Calibration** — apakah probabilitas bisa dipercaya?
- **Slicing** — performa per subkelompok (region/segmen) untuk cek fairness/robustness.
- **Error analysis** — baca kasus salah, cari pola.

---

## 11. Langkah 8 — Interpretasi & Model Card

```python
import shap
explainer = shap.Explainer(best_model.named_steps["clf"], ...)
# fitur apa yang menggerakkan prediksi? (global + lokal) — Modul 09
```

**Model Card** (dokumen wajib di tim serius) — ringkas:
- Tujuan & konteks penggunaan model.
- Data: sumber, periode, ukuran, keterbatasan.
- Metrik: performa di holdout + per-slice.
- Asumsi & batasan (di mana model TIDAK boleh dipakai).
- Pertimbangan etis/bias.
- Versi, tanggal, pemilik.

---

## 12. Langkah 9 — Persist Pipeline + Metadata

Simpan **seluruh pipeline** (preprocessing + model), bukan model telanjang — agar input produksi diproses identik dengan training.

```python
import joblib, json, datetime, sklearn

joblib.dump(best_model, "models/churn_pipeline_v1.joblib")

metadata = {
    "model_version": "1.0.0",
    "created_at": datetime.datetime.now().isoformat(),
    "git_commit": "...",                       # lacak kode yang menghasilkannya
    "sklearn_version": sklearn.__version__,    # versi penting untuk kompatibilitas
    "features": list(X.columns),
    "threshold": threshold,
    "metrics_holdout": {"pr_auc": ..., "recall": ...},
    "train_data_period": "2025-01..2025-12",
    "seed": SEED,
}
with open("models/churn_pipeline_v1.meta.json", "w") as f:
    json.dump(metadata, f, indent=2)
```
Lebih baik lagi: catat ke **MLflow** ([Modul 07](../07-mlops-deployment/README.md)) untuk versioning & perbandingan eksperimen.

---

## 13. Langkah 10 — Serah Terima ke Serving

Pipeline tersimpan langsung bisa dimuat di API ([Modul 07](../07-mlops-deployment/README.md)):

```python
model = joblib.load("models/churn_pipeline_v1.joblib")
# di FastAPI /predict: model.predict_proba(df_input)[:, 1] >= threshold
```
Pastikan rencana: **monitoring** (drift, performa), **logging prediksi**, **rollback**, dan **re-training** sudah didefinisikan sebelum go-live.

---

## 14. Checklist "Production-Approved" & Jebakan

### ✅ Checklist sebelum deploy
- [ ] Seed tetap; versi library & commit kode tercatat → **reproducible**.
- [ ] Test set dipisah **di awal**, disentuh **sekali**.
- [ ] Semua transform di dalam **Pipeline** (tak ada fit di luar fold train).
- [ ] Tidak ada fitur leakage (identifier, info masa depan, target turunan).
- [ ] Metrik **selaras tujuan bisnis**; threshold dari validation.
- [ ] Mengalahkan **baseline** secara meyakinkan.
- [ ] Evaluasi per-slice (fairness/robustness) + calibration + error analysis.
- [ ] Pipeline **lengkap** (prep+model) tersimpan + **metadata** + **model card**.
- [ ] Rencana monitoring, logging, rollback, re-training jelas.
- [ ] Unit test untuk transform & validasi data.

### 🕳️ Jebakan paling umum (penyebab "gagal di produksi")
1. **Data leakage** — #1 pembunuh. Scaling sebelum split, fitur dari masa depan, target encoding tanpa CV.
2. **Tuning di test set** — skor terlihat hebat, gagal nyata.
3. **Train-serving skew** — preprocessing di training beda dengan di serving (solusi: satu Pipeline).
4. **Metrik salah** — accuracy pada data imbalance.
5. **Tidak reproducible** — tanpa seed/versi, hasil tak bisa dipercaya/diulang.
6. **Kategori baru di produksi** bikin crash (solusi: `handle_unknown="ignore"`).
7. **Tidak ada baseline** — tak tahu apakah model menambah nilai.
8. **Model "membusuk" diam-diam** — tanpa monitoring drift, performa turun tak terdeteksi.

---

## Latihan
1. Jalankan [pipeline.py](./pipeline.py) end-to-end. Pahami tiap tahap & output (model, metadata, laporan).
2. Sisipkan leakage sengaja (scaling sebelum split) dan tunjukkan skor CV jadi menipu dibanding versi Pipeline yang benar.
3. Ganti model (LogReg → RandomForest/XGBoost) hanya dengan menukar langkah `clf` di Pipeline. Bandingkan vs baseline.
4. Tambah evaluasi per-slice & calibration plot ke laporan.
5. Buat **model card** lengkap untuk modelmu.
6. Muat pipeline tersimpan ke API FastAPI ([Modul 07](../07-mlops-deployment/README.md)) dan uji prediksi end-to-end.

⬅️ Kembali ke [Daftar Modul](../README.md) · Capstone ini menyatukan Modul **01, 02, 03, 07, 09, 10, 11, 12**.
