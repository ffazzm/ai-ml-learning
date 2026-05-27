# 00.2 · Aljabar Linear

> Aljabar linear adalah **bahasa data**. Sebuah gambar, kalimat, atau baris tabel — semuanya direpresentasikan sebagai **vektor** atau **matriks**. Operasi neural network = serangkaian perkalian matriks.

## 1. Skalar, Vektor, Matriks, Tensor

| Objek | Dimensi | Contoh | Notasi |
|---|---|---|---|
| **Skalar** | 0D | `5` | $x$ |
| **Vektor** | 1D | `[1, 2, 3]` | $\mathbf{v}$ |
| **Matriks** | 2D | tabel angka | $\mathbf{A}$ |
| **Tensor** | 3D+ | batch gambar (N×H×W×C) | $\mathcal{T}$ |

```python
import numpy as np

skalar = np.array(5)              # ndim 0
vektor = np.array([1, 2, 3])      # ndim 1, shape (3,)
matriks = np.array([[1, 2],
                    [3, 4]])      # ndim 2, shape (2, 2)
tensor = np.random.randn(2, 3, 4) # ndim 3
```

**Intuisi:** satu sampel data (mis. seseorang dengan fitur [umur, gaji, tinggi]) adalah **vektor**. Seluruh dataset (banyak orang × banyak fitur) adalah **matriks** $\mathbf{X}$ berukuran (n_sampel × n_fitur).

---

## 2. Operasi Vektor

### Penjumlahan & perkalian skalar

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b        # [5 7 9]
3 * a        # [3 6 9]
```

### Dot product (perkalian titik) — operasi terpenting

$$\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i = a_1 b_1 + a_2 b_2 + \dots + a_n b_n$$

```python
np.dot(a, b)     # 1*4 + 2*5 + 3*6 = 32
a @ b            # sama, operator @
```

**Mengapa penting?** Sebuah neuron menghitung `dot(input, weights) + bias`. Dot product mengukur **seberapa selaras** dua vektor. Ini juga dasar dari **cosine similarity** (kemiripan teks/embedding):

$$\text{cos\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}$$

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cosine_similarity(np.array([1, 0]), np.array([1, 1]))   # ~0.707
```

### Norm (panjang vektor)

$$\|\mathbf{v}\|_2 = \sqrt{\sum_i v_i^2}$$

```python
np.linalg.norm(a)       # norm L2 (Euclidean) = panjang vektor
np.linalg.norm(a, 1)    # norm L1 = jumlah nilai absolut
```

Norm dipakai untuk: mengukur jarak, **regularisasi** (L1/L2 di Modul 03), dan normalisasi.

---

## 3. Operasi Matriks

### Perkalian matriks (matmul)

Ini berbeda dari perkalian elemen. Untuk $\mathbf{C} = \mathbf{A}\mathbf{B}$:

$$C_{ij} = \sum_k A_{ik} B_{kj}$$

**Aturan shape:** $(m \times n) \times (n \times p) = (m \times p)$. Kolom kiri **harus** sama dengan baris kanan.

```python
A = np.array([[1, 2],
              [3, 4]])        # (2, 2)
B = np.array([[5, 6],
              [7, 8]])        # (2, 2)

A @ B
# [[19 22]
#  [43 50]]
# Cek: C[0,0] = 1*5 + 2*7 = 19 ✓

A * B          # ❌ ini perkalian elemen (Hadamard), BUKAN matmul
```

> **Inti neural network:** satu layer melakukan `output = X @ W + b`, di mana `X` adalah input (batch × fitur), `W` adalah bobot (fitur × neuron). Memahami matmul = memahami forward pass.

### Transpose

Membalik baris↔kolom: $(\mathbf{A}^T)_{ij} = A_{ji}$.

```python
A.T            # transpose
# shape (m,n) -> (n,m)
```

### Matriks identitas & invers

```python
I = np.eye(3)              # identitas: A @ I = A
A_inv = np.linalg.inv(A)   # invers: A @ A_inv = I (jika ada)
```

Invers muncul di solusi closed-form regresi linear (Normal Equation, Modul 03).

---

## 4. Sistem Persamaan Linear & Regresi

Banyak masalah ML dapat ditulis sebagai $\mathbf{X}\mathbf{w} = \mathbf{y}$ — cari bobot $\mathbf{w}$.

```python
# Selesaikan Xw = y (lebih stabil daripada inv(X) @ y):
w = np.linalg.solve(X, y)

# Untuk sistem over-determined (lebih banyak persamaan dari variabel) -> least squares:
w, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
```

Ini secara harfiah adalah cara melatih regresi linear tanpa gradient descent.

---

## 5. Konsep Lanjutan (kenali namanya)

Kamu tidak perlu menguasai semua ini sekarang, tapi **kenali apa fungsinya** karena akan muncul:

- **Eigenvalue & eigenvector** — arah "alami" sebuah transformasi matriks. Dasar dari **PCA** (reduksi dimensi).
- **SVD (Singular Value Decomposition)** — memecah matriks jadi komponen. Dipakai di PCA, sistem rekomendasi, kompresi.
- **Determinan** — mengukur "skala volume" transformasi; menentukan apakah matriks bisa diinvers (det ≠ 0).

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
U, S, Vt = np.linalg.svd(A)
np.linalg.det(A)
```

---

## Ringkasan Intuisi

| Operasi | "Untuk apa" dalam ML |
|---|---|
| Dot product | Neuron, similarity, proyeksi |
| Matmul | Forward pass tiap layer |
| Transpose | Menyelaraskan shape, backprop |
| Norm | Jarak, regularisasi, normalisasi |
| Invers / solve | Closed-form regresi |
| Eigen / SVD | PCA, reduksi dimensi, rekomendasi |

## Latihan

1. Implementasikan perkalian matriks dengan **triple loop** murni Python, lalu bandingkan hasilnya dengan `A @ B`. Rasakan kenapa NumPy diperlukan (coba untuk matriks 200×200, ukur waktu dengan `time`).
2. Diberi dua vektor embedding kata, hitung cosine similarity. Verifikasi: vektor yang sama → 1.0, vektor tegak lurus → 0.0.
3. Buat matriks data 100×3 acak. Hitung rata-rata tiap kolom **menggunakan matmul** dengan vektor `np.ones(100)/100`. (Petunjuk: `ones @ X / n`.)
4. Tonton seri "Essence of Linear Algebra" oleh 3Blue1Brown — wajib untuk intuisi visual.

➡️ Lanjut: [00.3 · Kalkulus & Gradien](./03-kalkulus-gradien.md)
