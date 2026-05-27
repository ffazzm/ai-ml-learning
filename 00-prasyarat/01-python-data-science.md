# 00.1 · Python untuk Data Science

Python adalah lingua franca AI/ML. Yang kita butuhkan bukan Python "umum", tapi **Python untuk komputasi numerik & data**: NumPy dan Pandas. Bagian ini mengasumsikan kamu sudah tahu dasar (variabel, `if`, `for`, fungsi, list, dict).

---

## 1. Python "Pythonic" yang sering dipakai di ML

### List comprehension & generator

```python
# Daripada loop manual:
squares = []
for x in range(10):
    squares.append(x ** 2)

# Pythonic:
squares = [x ** 2 for x in range(10)]

# Dengan kondisi:
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]

# Generator (hemat memori, dievaluasi lazy):
gen = (x ** 2 for x in range(1_000_000))   # tidak menyimpan semua di memori
```

### Dictionary & enumerate / zip

```python
words = ["ai", "ml", "dl"]

# enumerate: dapatkan index + nilai sekaligus
for i, w in enumerate(words):
    print(i, w)

# zip: iterasi dua list paralel
labels = [0, 1, 1]
for w, y in zip(words, labels):
    print(w, "->", y)

# dict comprehension: bikin vocab {kata: index}
vocab = {w: i for i, w in enumerate(words)}
```

### Unpacking & `*args`, `**kwargs`

```python
a, b, *rest = [1, 2, 3, 4]    # a=1, b=2, rest=[3,4]

def train(model, *args, **kwargs):
    print(args)     # tuple posisi
    print(kwargs)   # dict keyword
```

> **Kenapa penting?** API library ML (scikit-learn, PyTorch) penuh dengan `**kwargs`, unpacking, dan comprehension. Membaca kode orang lain jadi jauh lebih mudah.

---

## 2. NumPy — fondasi semua komputasi numerik

NumPy menyediakan **ndarray**: array N-dimensi yang cepat (operasi di C, bukan loop Python). **Semua** data ML pada akhirnya adalah array NumPy (atau tensor, yang merupakan generalisasinya).

### Membuat array

```python
import numpy as np

a = np.array([1, 2, 3])                 # vektor (1D)
b = np.array([[1, 2], [3, 4]])          # matriks (2D)

np.zeros((2, 3))        # matriks 2x3 berisi 0
np.ones((2, 3))         # berisi 1
np.eye(3)               # matriks identitas 3x3
np.arange(0, 10, 2)     # [0 2 4 6 8]
np.linspace(0, 1, 5)    # 5 angka rata dari 0 sampai 1
np.random.randn(3, 3)   # 3x3 dari distribusi normal standar
```

### Atribut penting

```python
b.shape      # (2, 2)  -> bentuk
b.ndim       # 2       -> jumlah dimensi
b.dtype      # dtype('int64')
b.size       # 4       -> total elemen
```

> **`.shape` adalah teman terbaikmu.** 90% bug di ML berasal dari shape yang tidak cocok. Selalu `print(x.shape)` saat debugging.

### Vectorization — JANGAN pakai loop

```python
x = np.arange(1_000_000)

# ❌ Lambat (loop Python):
y = np.array([v * 2 + 1 for v in x])

# ✅ Cepat (vectorized, dieksekusi di C):
y = x * 2 + 1
```

Operasi diterapkan elemen-per-elemen secara otomatis. Ini bisa **100x lebih cepat**.

### Indexing & slicing

```python
a = np.array([10, 20, 30, 40, 50])
a[0]          # 10
a[-1]         # 50
a[1:4]        # [20 30 40]

m = np.array([[1, 2, 3],
              [4, 5, 6]])
m[0, 1]       # 2  (baris 0, kolom 1)
m[:, 0]       # [1 4]  (semua baris, kolom 0)
m[1, :]       # [4 5 6]  (baris 1, semua kolom)

# Boolean indexing (sangat sering dipakai):
a[a > 25]     # [30 40 50]
a[a > 25] = 0 # set semua >25 jadi 0
```

### Broadcasting — operasi antar shape berbeda

NumPy otomatis "merentangkan" array kecil agar cocok dengan yang besar.

```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape (2, 3)
v = np.array([10, 20, 30])     # shape (3,)

m + v
# [[11 22 33]
#  [14 25 36]]   <- v ditambahkan ke setiap baris
```

Aturan broadcasting: dimensi dibandingkan dari kanan; cocok jika **sama** atau salah satunya **1**. Ini dipakai di mana-mana, contoh: menambahkan bias ke setiap baris output neural network.

### Operasi aljabar & agregasi

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A * B          # perkalian elemen-per-elemen (Hadamard)
A @ B          # perkalian MATRIKS (dot product) <- beda!
A.T            # transpose

A.sum()              # total semua
A.sum(axis=0)        # jumlah per kolom -> [4 6]
A.sum(axis=1)        # jumlah per baris -> [3 7]
A.mean(), A.std(), A.max(), A.argmax()
```

> **`axis` itu "dimensi yang dihilangkan/diciutkan".** `axis=0` menciutkan baris (hasil per kolom). Hafalkan dengan latihan, ini sumber kebingungan klasik.

---

## 3. Pandas — manipulasi data tabular

Pandas adalah "Excel dengan steroid untuk programmer". Dua struktur inti: **Series** (1 kolom) dan **DataFrame** (tabel).

### Membaca & melihat data

```python
import pandas as pd

df = pd.read_csv("data.csv")

df.head()          # 5 baris pertama
df.tail(3)         # 3 baris terakhir
df.shape           # (jumlah_baris, jumlah_kolom)
df.info()          # tipe data & non-null count
df.describe()      # statistik ringkas kolom numerik
df.columns         # nama kolom
df.dtypes          # tipe tiap kolom
```

### Memilih data

```python
df["umur"]                       # satu kolom (Series)
df[["umur", "gaji"]]             # beberapa kolom (DataFrame)

df.loc[0]                        # baris berdasarkan LABEL index
df.iloc[0]                       # baris berdasarkan POSISI
df.loc[0, "umur"]                # sel spesifik

# Filtering (boolean mask):
df[df["umur"] > 30]
df[(df["umur"] > 30) & (df["gaji"] < 5000)]   # pakai & | bukan and/or
```

### Membersihkan & transformasi

```python
df.isnull().sum()                # hitung missing per kolom
df.dropna()                      # buang baris yang ada NaN
df["gaji"].fillna(df["gaji"].median())   # isi NaN dengan median

df["gaji_juta"] = df["gaji"] / 1_000_000      # kolom baru
df["umur"].apply(lambda x: "tua" if x > 40 else "muda")
df["kota"].map({"jkt": "Jakarta", "bdg": "Bandung"})

df.rename(columns={"gaji": "salary"})
df.drop(columns=["id"])
```

### Group by — agregasi (sangat penting untuk EDA)

```python
# Rata-rata gaji per kota:
df.groupby("kota")["gaji"].mean()

# Beberapa agregasi sekaligus:
df.groupby("kota").agg(
    gaji_rata=("gaji", "mean"),
    jumlah=("gaji", "count"),
)
```

### Dari Pandas ke NumPy (untuk model)

```python
X = df[["umur", "gaji"]].values     # -> ndarray, siap untuk scikit-learn
y = df["label"].values
```

---

## 4. Visualisasi cepat (Matplotlib)

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot([1, 2, 3], [1, 4, 9], label="kuadrat")
plt.scatter([1, 2, 3], [2, 3, 5])
plt.xlabel("x"); plt.ylabel("y")
plt.title("Contoh plot")
plt.legend()
plt.show()

# Histogram distribusi:
plt.hist(df["umur"], bins=20)
plt.show()
```

`seaborn` (`import seaborn as sns`) memberi plot statistik yang lebih cantik dengan kode lebih sedikit — kita pakai banyak di Modul 02.

---

## Latihan

1. Buat array NumPy `1..100`, lalu ambil hanya yang habis dibagi 7 — **tanpa loop**.
2. Buat matriks acak 5×5, normalisasi tiap **kolom** sehingga mean 0 dan std 1 (gunakan broadcasting & `axis`).
3. Unduh dataset CSV apa saja (mis. Titanic dari Kaggle). Hitung: berapa % data hilang per kolom? Rata-rata umur per kelas tiket? Buat histogram distribusi umur.
4. Tanpa menjalankan: apa hasil shape dari `np.ones((3,1)) + np.ones((1,4))`? (Jawab: `(3,4)` — pahami kenapa lewat broadcasting.)

➡️ Lanjut: [00.2 · Aljabar Linear](./02-aljabar-linear.md)
