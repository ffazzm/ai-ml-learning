# 00.5 · Penerapan Matematika & Statistika dalam Proses AI/ML

> Matematika di AI/ML **bukan topik terpisah** yang "selesai lalu dilupakan". Ia muncul di **setiap tahap** pipeline. Dokumen ini memetakan: konsep matematika/statistika mana yang bekerja di tahap mana — supaya kamu tahu *kenapa* mempelajarinya.

## Peta Ringkas

```
Tahap pipeline           Matematika yang bekerja
─────────────────────────────────────────────────────────
Data & EDA          →    Statistika deskriptif
Representasi data    →    Aljabar linear (vektor / matriks / tensor)
Forward pass        →    Aljabar linear (perkalian matriks)
Training / learning  →    Kalkulus (gradien, chain rule)
Loss function       →    Probabilitas (entropi, Bayes)
Evaluasi            →    Statistika inferensial (uji, CI, distribusi)
Interpretasi        →    Statistika + teori game (SHAP)
```

**Tiga pilar besarnya:**
- **Aljabar linear** = *bagaimana data direpresentasikan & dihitung*.
- **Kalkulus** = *bagaimana model belajar*.
- **Probabilitas & statistik** = *bagaimana kita mengukur ketidakpastian & menilai hasil*.

---

## 1. Tahap Data & EDA → Statistika Deskriptif

Sebelum modeling, kamu memahami data dengan statistik (lihat [Modul 02](../02-data-feature-engineering/README.md)).

| Konsep | Dipakai untuk |
|---|---|
| Mean, median, modus | imputasi missing value (median tahan outlier) |
| Variansi & std | ukur sebaran; dasar standardisasi fitur |
| Kuartil & IQR | deteksi outlier (`Q1 − 1.5·IQR`) |
| Korelasi (Pearson) | cari fitur redundan & fitur yang berhubungan dengan target |
| Skewness / distribusi | putuskan perlu transformasi (log) atau tidak |

```python
import numpy as np
gaji = df["gaji"]
print(gaji.median())                 # imputasi pakai median krn distribusi miring
batas = gaji.quantile(0.75) + 1.5 * (gaji.quantile(0.75) - gaji.quantile(0.25))
outlier = gaji[gaji > batas]         # deteksi outlier via IQR
df["gaji_log"] = np.log1p(gaji)      # normalkan distribusi yang skewed
```

> Keputusan "pakai median, bukan mean" adalah keputusan **statistik**, bukan sekadar pilihan kode.

---

## 2. Tahap Representasi Data → Aljabar Linear

Semua data pada akhirnya jadi vektor/matriks ([Modul 00.2](./02-aljabar-linear.md)).

- 1 sampel = **vektor**; seluruh dataset = **matriks** `X` berukuran (n_sampel × n_fitur).
- Gambar = **tensor** (C×H×W); teks = vektor **embedding**.
- **Cosine similarity** (dot product ternormalisasi) → mesin semantic search & RAG ([Modul 05](../05-nlp-llm/README.md)).
- **PCA** (reduksi dimensi) memakai **eigenvalue / SVD**.

```python
def cosine_similarity(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
# dipakai untuk mengukur kemiripan makna antar embedding kalimat
```

---

## 3. Tahap "Model Menghitung" (Forward Pass) → Aljabar Linear

Inti tiap model = operasi matriks.

- Regresi linear: `ŷ = X @ w + b` — **perkalian matriks**.
- Tiap layer neural network: `output = aktivasi(X @ W + b)`.
- Satu neuron = **dot product** input dengan bobot.

```python
# Satu layer neural network = satu perkalian matriks
output = X @ W + b        # X:(batch, fitur)  W:(fitur, neuron)  -> (batch, neuron)
```

---

## 4. Tahap "Belajar" (Training) → Kalkulus

Jantung dari kata *learning*. Model memperbaiki diri dengan menuruni loss ([Modul 00.3](./03-kalkulus-gradien.md)).

- **Turunan/gradien** → arah memperbaiki parameter.
- **Gradient descent**: `θ ← θ − α·∇L`.
- **Chain rule** → backpropagation pada neural network ([Modul 04](../04-deep-learning/README.md)).
- **Optimizer** (Adam, SGD) = variasi cara memakai gradien.

```python
for step in range(n_iter):
    grad = compute_gradient(loss, params)   # kalkulus: ∇L
    params = params - lr * grad             # langkah turun melawan gradien
```

> Setiap kali loss turun saat training, itu **kalkulus** yang bekerja. Tanpanya, tidak ada "learning".

---

## 5. Tahap Loss Function → Probabilitas

Cara mengukur "salah" berakar di probabilitas ([Modul 00.4](./04-probabilitas-statistik.md)).

- **Cross-entropy** (loss klasifikasi) = jarak antar dua distribusi probabilitas; berasal dari **entropi**.
- **MSE** (loss regresi) = berhubungan dengan asumsi error berdistribusi normal.
- **Softmax / sigmoid** = mengubah skor mentah jadi probabilitas sah (jumlah = 1).
- **Naive Bayes** = penerapan langsung **Teorema Bayes**.

```python
def cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred))   # loss = "kekagetan" prediksi salah
```

---

## 6. Tahap Evaluasi → Statistika Inferensial

Menilai model dengan jujur ([Modul 01](../01-fondasi-ml/README.md) & [Modul 09](../09-evaluasi-interpretasi/README.md)).

| Konsep statistik | Peran dalam evaluasi |
|---|---|
| Sampel vs populasi | dasar train/test split & generalisasi |
| Cross-validation | estimasi performa yang andal (rata-rata banyak fold) |
| Precision/recall/AUC | dihitung dari probabilitas kondisional `P(benar\|prediksi)` |
| Confidence interval | seberapa yakin skor bukan kebetulan |
| Uji hipotesis (KS, t-test) | deteksi **data drift** ([Modul 07](../07-mlops-deployment/README.md)), bandingkan model A/B |
| Bias-variance | dekomposisi error secara statistik |

```python
from scipy.stats import ks_2samp
stat, p = ks_2samp(fitur_train, fitur_produksi)   # uji statistik untuk deteksi drift
if p < 0.05:
    print("Distribusi berubah signifikan → pertimbangkan re-training")
```

---

## 7. Tahap Interpretasi → Statistika + Teori Game

Menjelaskan *kenapa* model memutuskan begitu ([Modul 09](../09-evaluasi-interpretasi/README.md)).

- **Permutation importance** → uji statistik: acak satu fitur, ukur penurunan performa.
- **SHAP** → berbasis **Shapley value** (teori game) + ekspektasi: bagi kontribusi tiap fitur secara adil.
- **Calibration** → bandingkan probabilitas prediksi vs frekuensi empiris nyata.

---

## Penutup

Matematika di AI/ML bukan "gerbang ujian" yang dilewati sekali. Ia hadir di **setiap baris** pipeline:

1. **Statistika** memandumu memahami & membersihkan data, lalu menilai model.
2. **Aljabar linear** adalah cara data hidup di dalam mesin & dihitung.
3. **Kalkulus** adalah mesin yang membuat model benar-benar belajar.
4. **Probabilitas** menyatukan ketidakpastian, loss, dan banyak algoritma.

Itulah kenapa [Modul 00](./README.md) menaruh keempatnya di fondasi — bukan untuk dihafal, tapi untuk dikenali saat muncul di modul-modul berikutnya.

⬅️ Kembali ke [Modul 00 · Prasyarat](./README.md)
