# Modul 01 · Fondasi Machine Learning

> Sebelum menyentuh algoritma, kamu harus paham **kerangka berpikir** ML: apa yang sebenarnya terjadi saat "mesin belajar", dan bagaimana kita tahu modelnya bagus. Modul ini adalah konsep yang akan dipakai di **setiap** modul setelahnya.

## Tujuan Belajar
- Mendefinisikan ML dan membedakan jenis-jenisnya.
- Memahami workflow ML end-to-end.
- Menguasai konsep **overfitting, underfitting, bias-variance**.
- Bisa memilih & menafsirkan **metrik evaluasi** yang tepat.
- Paham kenapa kita **split data** dan melakukan **cross-validation**.

---

## 1. Apa itu Machine Learning?

> "Machine Learning adalah bidang studi yang memberi komputer kemampuan belajar tanpa diprogram secara eksplisit." — Arthur Samuel (1959)

**Pemrograman tradisional:** kamu menulis aturan.
```
Aturan + Data  →  Jawaban
```

**Machine Learning:** kamu memberi contoh, mesin menemukan aturannya.
```
Data + Jawaban  →  Aturan (model)
```

**Contoh:** untuk deteksi spam, kamu **tidak** menulis ribuan `if "menang lotre" in email`. Kamu beri 10.000 email berlabel spam/bukan, dan model **menemukan sendiri** pola pembeda.

### Kapan pakai ML (dan kapan tidak)
✅ Cocok: pola kompleks, aturan sulit ditulis manual, data banyak, masalah berulang (deteksi penipuan, rekomendasi, prediksi harga).
❌ Tidak perlu: aturan sederhana & jelas (validasi format email pakai regex saja), data sedikit/tidak ada, butuh penjelasan 100% deterministik.

---

## 2. Jenis-Jenis Machine Learning

### a) Supervised Learning (belajar dengan label)
Data punya **input (X)** dan **jawaban benar (y)**. Model belajar memetakan X → y.

- **Regresi** — prediksi nilai kontinu. *Contoh: harga rumah, suhu besok.*
- **Klasifikasi** — prediksi kategori. *Contoh: spam/bukan, jenis penyakit, sentimen.*

### b) Unsupervised Learning (tanpa label)
Hanya ada **input (X)**, tidak ada jawaban. Model mencari struktur tersembunyi.

- **Clustering** — kelompokkan data mirip. *Contoh: segmentasi pelanggan.*
- **Dimensionality Reduction** — sederhanakan fitur. *Contoh: PCA untuk visualisasi.*
- **Association** — temukan aturan "yang beli A juga beli B".

### c) Reinforcement Learning (belajar dari reward)
Agent berinteraksi dengan lingkungan, belajar dari **reward/punishment**. *Contoh: AlphaGo, robotika, game.* (Di luar fokus utama kurikulum ini, tapi penting dikenali.)

### d) Self-supervised Learning
Label dibuat otomatis dari data itu sendiri. *Contoh: LLM memprediksi kata berikutnya.* Ini fondasi model modern seperti GPT (Modul 05).

```
                    Machine Learning
                          │
      ┌───────────┬───────┴────────┬──────────────┐
  Supervised   Unsupervised   Reinforcement   Self-supervised
   (ada y)      (tanpa y)      (reward)        (label otomatis)
   ┌────┴────┐   ┌────┴─────┐
 Regresi  Klasifikasi  Clustering  Dim.Reduction
```

---

## 3. Workflow ML End-to-End

Project ML nyata adalah siklus, bukan garis lurus:

```
1. Definisikan masalah    → Apa yang diprediksi? Metrik sukses bisnis apa?
2. Kumpulkan data         → Sumber, kualitas, jumlah
3. EDA & preprocessing    → Pahami & bersihkan (Modul 02)
4. Feature engineering    → Buat fitur informatif (Modul 02)
5. Pilih & latih model    → Mulai sederhana (Modul 03/04)
6. Evaluasi               → Metrik di data yang belum dilihat
7. Tuning hyperparameter  → Perbaiki performa
8. Deploy                 → Layani prediksi (Modul 07)
9. Monitor & iterasi      → Data berubah, model perlu diperbarui
        └──────────────── kembali ke atas ──────────────┘
```

> **Aturan emas:** ~80% waktu project ML habis di langkah 2–4 (data), bukan modeling. "Garbage in, garbage out."

---

## 4. Konsep Inti: Generalisasi

Tujuan ML **bukan** menghafal data training, tapi **bekerja baik pada data baru** yang belum pernah dilihat. Ini disebut **generalisasi**.

### Train / Validation / Test split
Kita **wajib** memisahkan data:

| Set | Fungsi | Porsi umum |
|---|---|---|
| **Training** | model belajar dari sini | 60–80% |
| **Validation** | tuning hyperparameter & pilih model | 10–20% |
| **Test** | evaluasi akhir, **disentuh sekali** | 10–20% |

```python
from sklearn.model_selection import train_test_split

# Pisah dulu test (jangan pernah disentuh sampai akhir)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Lalu pisah train & validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)
```

> 🚨 **Data leakage** = dosa terbesar di ML. Jika informasi dari test bocor ke training (mis. melakukan scaling sebelum split), evaluasimu jadi bohong dan model gagal di produksi. Selalu split **dulu**, baru preprocessing pakai statistik dari train saja.

---

## 5. Overfitting vs Underfitting

| | Underfitting | Pas (Good Fit) | Overfitting |
|---|---|---|---|
| **Gejala** | buruk di train & test | bagus di keduanya | bagus di train, buruk di test |
| **Penyebab** | model terlalu sederhana | — | model terlalu kompleks / data kurang |
| **Analogi** | murid tak belajar | murid paham konsep | murid menghafal soal tanpa paham |

```
   Underfitting          Good Fit            Overfitting
   (high bias)                              (high variance)
      ___                  _/\_                  /\  /\
     /                    /    \               _/  \/  \_
  __/        garis     __/      \__         (lika-liku ikut tiap titik)
  terlalu lurus        pola pas             menghafal noise
```

### Cara mengatasi overfitting
- Tambah data training.
- **Regularisasi** (L1/L2 — penalti untuk bobot besar; Modul 03).
- Kurangi kompleksitas model (lebih sedikit fitur/parameter).
- **Dropout**, **early stopping** (Modul 04).
- Cross-validation untuk deteksi dini.

### Cara mengatasi underfitting
- Model lebih kompleks / lebih banyak fitur.
- Latih lebih lama, kurangi regularisasi.

---

## 6. Bias-Variance Tradeoff

Total error model = **Bias² + Variance + Noise** (irreducible).

- **Bias tinggi** = asumsi terlalu kaku → underfitting.
- **Variance tinggi** = terlalu sensitif pada data spesifik → overfitting.
- Menurunkan satu sering menaikkan yang lain → **tradeoff**. Tujuan: titik keseimbangan dengan total error minimum.

```
error
  │\                              /
  │ \  variance ___________----/
  │  \________            __--
  │  bias    \______ __--/   ← total error (cari minimum di sini)
  │                X
  └───────────────────────────── kompleksitas model →
```

---

## 7. Metrik Evaluasi (sangat penting!)

**Akurasi saja sering menyesatkan.** Pilih metrik sesuai masalah.

### Untuk Klasifikasi — Confusion Matrix
```
                 Prediksi Positif   Prediksi Negatif
Aktual Positif      TP (benar)        FN (lolos)
Aktual Negatif      FP (alarm palsu)  TN (benar)
```

| Metrik | Rumus | Pakai saat |
|---|---|---|
| **Accuracy** | (TP+TN)/total | kelas seimbang |
| **Precision** | TP/(TP+FP) | FP mahal (mis. email penting masuk spam) |
| **Recall** | TP/(TP+FN) | FN mahal (mis. lolos deteksi kanker) |
| **F1-score** | 2·(P·R)/(P+R) | butuh keseimbangan P & R |
| **ROC-AUC** | area bawah kurva | ranking, kelas tidak seimbang |

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))   # precision, recall, f1 sekaligus
print("AUC:", roc_auc_score(y_test, y_proba))  # y_proba = probabilitas, bukan label
```

> **Contoh kelas tidak seimbang:** deteksi penipuan dengan 99% transaksi normal. Model yang selalu bilang "normal" punya **akurasi 99%** tapi **tidak berguna** (recall penipuan = 0). Inilah kenapa precision/recall wajib.

### Untuk Regresi

| Metrik | Arti |
|---|---|
| **MAE** | rata-rata selisih absolut — mudah ditafsir, tahan outlier |
| **MSE** | rata-rata selisih kuadrat — menghukum error besar |
| **RMSE** | √MSE — satuan sama dengan target |
| **R²** | proporsi variansi yang dijelaskan (1 = sempurna, 0 = sebaik menebak mean) |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
```

---

## 8. Cross-Validation

Daripada satu split tetap (yang bisa "beruntung/sial"), **k-fold CV** membagi data jadi k bagian, melatih k kali (tiap kali 1 fold jadi validasi), lalu rata-ratakan. Estimasi performa jadi lebih andal.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

scores = cross_val_score(LogisticRegression(), X, y, cv=5, scoring="f1")
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

```
Fold:  [test][train][train][train][train]   → skor 1
       [train][test][train][train][train]   → skor 2
       ...                                    → rata-rata = estimasi andal
```

---

## Ringkasan & Mental Model

1. ML = belajar pola dari data untuk **generalisasi** ke data baru.
2. Selalu **split data**; jaga dari **leakage**.
3. Waspadai **overfitting/underfitting**; pahami **bias-variance**.
4. Pilih **metrik** sesuai konteks bisnis — akurasi bukan segalanya.
5. Pakai **cross-validation** untuk evaluasi yang andal.

## Latihan
1. Jelaskan dengan kata-katamu sendiri kenapa kita tidak boleh memilih model berdasarkan performa di data test.
2. Untuk tiap kasus, metrik apa yang utama: (a) deteksi tumor, (b) filter spam, (c) rekomendasi film, (d) prediksi harga saham. Beri alasan.
3. Buat dataset sintetis, latih model sangat kompleks (mis. polynomial degree tinggi), dan tunjukkan overfitting dengan membandingkan error train vs test.
4. Implementasikan k-fold cross-validation **dari nol** dengan NumPy (tanpa scikit-learn) untuk memahami mekanismenya.

➡️ Lanjut: [Modul 02 · Data & Feature Engineering](../02-data-feature-engineering/README.md)
