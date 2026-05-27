# Modul 13 · Performa & Concurrency Python untuk AI Engineering

> Model yang benar tapi lambat sering tak berguna di produksi. Modul ini soal **menulis Python yang cepat & scalable**: kenapa loop lambat, bagaimana vectorization mempercepat 100×, dan kapan memakai threading vs multiprocessing vs async. Ini skill engineering yang memisahkan AI Engineer dari sekadar "yang bisa training".

## Tujuan Belajar
- Memahami **kenapa Python lambat** & bagaimana NumPy/PyTorch mengatasinya.
- Menguasai **vectorization** dan menghindari loop Python.
- Memahami **memory**: view vs copy, dtype, kontiguitas.
- Melakukan **profiling** untuk menemukan bottleneck.
- Memahami **GIL** dan memilih **threading / multiprocessing / asyncio** dengan benar.
- Menerapkan pola paralel nyata: data loading, batch inference, panggilan API LLM massal.

## Daftar Isi
1. Kenapa Python lambat (dan kenapa AI tetap pakai Python)
2. Vectorization vs for loop ⭐
3. Teknik vectorization lanjutan
4. Memory: view, copy, dtype, kontiguitas
5. Profiling — ukur dulu, baru optimasi
6. GIL: kunci pemahaman concurrency
7. Threading (I/O-bound)
8. Multiprocessing (CPU-bound)
9. Asyncio (banyak I/O konkuren — mis. API LLM)
10. Pola paralel dalam ML/AI nyata
11. JIT & akselerasi (Numba, Cython)
12. Cheat sheet & latihan

> 📂 Benchmark yang bisa dijalankan: [benchmark.py](./benchmark.py) — bandingkan loop vs vectorization vs multiprocessing dengan angka nyata.

---

## 1. Kenapa Python Lambat (dan Kenapa AI Tetap Pakai Python)

Python itu **interpreted** & **dynamically typed** — tiap operasi mengecek tipe saat runtime, dan loop dieksekusi baris demi baris oleh interpreter. Ini ~10–100× lebih lambat dari C untuk komputasi numerik.

**Rahasianya:** library AI (NumPy, PyTorch, TensorFlow) hanyalah "pembungkus" Python tipis di atas kode **C/C++/CUDA** yang sangat cepat. Saat kamu menulis `A @ B`, perkaliannya terjadi di C, bukan Python. Kamu menulis kode tingkat tinggi yang nyaman, mesin menjalankannya cepat.

> Prinsip inti AI Engineering: **biarkan komputasi berat terjadi di dalam library (C/GPU), bukan di loop Python.**

---

## 2. Vectorization vs For Loop ⭐

Vectorization = melakukan operasi pada **seluruh array sekaligus** (di C) alih-alih elemen per elemen (loop Python).

```python
import numpy as np
import time

x = np.arange(10_000_000)

# ❌ Loop Python — lambat
start = time.perf_counter()
hasil = []
for v in x:
    hasil.append(v * 2 + 1)
print(f"Loop:         {time.perf_counter() - start:.3f}s")

# ✅ Vectorized — operasi di C
start = time.perf_counter()
hasil = x * 2 + 1
print(f"Vectorized:   {time.perf_counter() - start:.3f}s")
# Biasanya 50-200× lebih cepat!
```

### Kenapa jauh lebih cepat?
1. Loop dijalankan di C, bukan interpreter Python.
2. Data tersimpan rapat di memori (tipe seragam) → ramah cache CPU.
3. CPU memakai instruksi **SIMD** (proses banyak angka sekaligus).

### Aturan emas
> Kalau kamu menulis `for` untuk mengiterasi array NumPy elemen-per-elemen, **hampir pasti ada cara vectorized yang lebih baik.**

```python
# Contoh: terapkan kondisi tanpa loop
x = np.array([1, -2, 3, -4, 5])
np.where(x > 0, x, 0)        # ReLU: ganti negatif jadi 0 -> [1 0 3 0 5]

# Agregasi tanpa loop
x.sum(); x.mean(); x.max(axis=0)

# Operasi antar array (broadcasting, lihat Modul 00.1)
A = np.random.randn(1000, 50)
b = np.random.randn(50)
A + b                        # b ditambahkan ke setiap baris, tanpa loop
```

---

## 3. Teknik Vectorization Lanjutan

```python
# Boolean masking — filter & assign tanpa loop
data = np.random.randn(1_000_000)
data[data < 0] = 0                          # set semua negatif jadi 0
positif = data[data > 0.5]                  # ambil subset

# Fancy indexing
idx = np.array([0, 5, 10])
data[idx]                                   # ambil banyak indeks sekaligus

# np.einsum — operasi tensor kompleks dalam satu ekspresi (kuat untuk DL)
A = np.random.randn(100, 200)
B = np.random.randn(200, 300)
np.einsum("ij,jk->ik", A, B)                # = A @ B, tapi bisa lebih kompleks

# Vektorisasi fungsi kustom (jika tak ada padanan NumPy)
def f(x):
    return x ** 2 if x > 0 else 0
f_vec = np.vectorize(f)                     # nyaman TAPI tetap lambat (loop tersembunyi)
# -> lebih baik tulis ulang dengan np.where bila bisa
```

> ⚠️ `np.vectorize` & `.apply()` Pandas itu **bukan** vectorization sejati — masih loop di balik layar. Cari operasi NumPy/Pandas native dulu.

### Di Pandas
```python
# ❌ Lambat
df["c"] = df.apply(lambda row: row["a"] + row["b"], axis=1)
# ✅ Vectorized
df["c"] = df["a"] + df["b"]
```

---

## 4. Memory: View, Copy, Dtype, Kontiguitas

Performa bukan cuma kecepatan CPU — **memori** sering jadi bottleneck untuk data besar.

### View vs Copy (penting & sering bikin bug)
```python
a = np.arange(10)
b = a[2:5]          # VIEW — berbagi memori dengan a (tak menyalin)
b[0] = 999          # mengubah a juga!  a[2] sekarang 999

c = a[2:5].copy()   # COPY — independen
```
Slicing menghasilkan **view** (hemat memori, tapi hati-hati efek samping). Boolean/fancy indexing menghasilkan **copy**.

### Dtype — pilih tipe yang tepat
```python
x = np.ones(1_000_000, dtype=np.float64)    # 8 MB
y = np.ones(1_000_000, dtype=np.float32)    # 4 MB — setengahnya!

# Untuk deep learning, float32 standar (float16/bfloat16 untuk hemat memori GPU)
df["kategori"] = df["kategori"].astype("category")   # Pandas: hemat memori drastis
```
Memilih `float32` vs `float64`, atau `int8` vs `int64`, bisa memangkas memori 2–8× — krusial untuk dataset & model besar.

### Generator — proses data besar tanpa muat semua ke RAM
```python
# ❌ Muat 10 juta angka ke memori
data = [x ** 2 for x in range(10_000_000)]

# ✅ Generator — dievaluasi lazy, satu per satu (hemat memori)
data = (x ** 2 for x in range(10_000_000))

# Baca file besar baris demi baris tanpa load seluruhnya
def baca_baris(path):
    with open(path) as f:
        for baris in f:          # streaming
            yield baris.strip()
```

---

## 5. Profiling — Ukur Dulu, Baru Optimasi

> "Premature optimization is the root of all evil." Jangan tebak bottleneck — **ukur**.

```python
# Di Jupyter/IPython:
%timeit np.sum(x)               # waktu rata-rata satu baris
%%timeit                        # waktu seluruh sel
...

# cProfile — temukan fungsi mana yang makan waktu
import cProfile
cProfile.run("train_model()")

# line_profiler — waktu per baris (pip install line_profiler)
# @profile decorator + kernprof -l -v script.py

# Memori:
# memory_profiler (pip install memory_profiler) -> @profile + mprof
```

Alur: profil → temukan 1–2 bottleneck terbesar → optimasi yang itu saja → ukur lagi. 80% waktu biasanya di 20% kode.

---

## 6. GIL — Kunci Memahami Concurrency Python

**GIL (Global Interpreter Lock)** = mekanisme CPython yang membuat **hanya satu thread menjalankan bytecode Python pada satu waktu**. Konsekuensinya besar:

- **Multithreading TIDAK mempercepat kode CPU-bound murni Python** (thread bergantian, bukan paralel sejati).
- Tapi GIL **dilepas** saat: operasi I/O (baca file, network) dan saat di dalam kode C (NumPy, PyTorch).

Maka pilihan tool tergantung jenis pekerjaan:

| Jenis pekerjaan | Contoh | Pakai |
|---|---|---|
| **I/O-bound** | unduh file, panggil API, query DB, baca disk | **threading** atau **asyncio** |
| **CPU-bound (Python murni)** | loop perhitungan, parsing teks berat | **multiprocessing** |
| **CPU-bound (NumPy/PyTorch)** | matmul, training | sudah paralel di C/GPU — cukup vectorize |

> Catatan: Python 3.13+ mulai punya mode "no-GIL" eksperimental, tapi prinsip di atas masih panduan praktis saat ini.

---

## 7. Threading — untuk I/O-bound

Banyak thread menunggu I/O secara bergantian. Karena GIL dilepas saat menunggu I/O, ini **efektif untuk tugas I/O** (mis. unduh 100 file, panggil banyak API).

```python
from concurrent.futures import ThreadPoolExecutor
import requests

urls = [...]   # banyak URL

def unduh(url):
    return requests.get(url).status_code

# Tanpa threading: unduh berurutan (lambat, banyak waktu menunggu)
# Dengan threading: banyak unduhan menunggu bersamaan
with ThreadPoolExecutor(max_workers=20) as ex:
    hasil = list(ex.map(unduh, urls))
```

---

## 8. Multiprocessing — untuk CPU-bound

Membuat **proses terpisah** (tiap proses punya interpreter & GIL sendiri) → paralel sejati di banyak core CPU. Cocok untuk komputasi Python murni yang berat.

```python
from concurrent.futures import ProcessPoolExecutor
import math

def kerja_berat(n):
    return sum(math.sqrt(i) for i in range(n))   # CPU-bound Python murni

tugas = [10_000_000] * 8
with ProcessPoolExecutor(max_workers=4) as ex:
    hasil = list(ex.map(kerja_berat, tugas))     # tersebar ke 4 core
```

> ⚠️ Multiprocessing punya **overhead**: memulai proses & menyalin data antar proses (pickling) mahal. Hanya untung kalau tiap tugas cukup berat. Untuk data besar, hindari mengirim array raksasa bolak-balik.

### joblib — dipakai scikit-learn di balik layar
```python
from joblib import Parallel, delayed

hasil = Parallel(n_jobs=-1)(delayed(kerja_berat)(n) for n in tugas)
# n_jobs=-1 -> pakai semua core. Ini yang dipakai cross_val_score, GridSearchCV (n_jobs=-1)
```

---

## 9. Asyncio — Banyak I/O Konkuren (mis. API LLM)

Untuk **ribuan operasi I/O konkuren** (panggilan API, request web), `asyncio` lebih ringan dari threading (satu thread, banyak tugas bergantian secara kooperatif). **Sangat relevan untuk AI Engineer** yang memanggil API LLM/embedding dalam jumlah besar.

```python
import asyncio
import aiohttp

async def panggil_api(session, payload):
    async with session.post("https://api.contoh.com", json=payload) as resp:
        return await resp.json()

async def main(daftar_payload):
    async with aiohttp.ClientSession() as session:
        tugas = [panggil_api(session, p) for p in daftar_payload]
        return await asyncio.gather(*tugas)        # jalankan semua konkuren

# hasil = asyncio.run(main(daftar_payload))
```

> Contoh nyata: memproses 10.000 dokumen lewat API embedding. Berurutan = berjam-jam; async dengan batas konkurensi = menit. Selalu pasang **batas konkurensi** (`asyncio.Semaphore`) & **retry** agar tidak kena rate limit.

---

## 10. Pola Paralel dalam ML/AI Nyata

| Tugas | Solusi |
|---|---|
| Loading data saat training | **PyTorch DataLoader** `num_workers>0` (multiprocessing) |
| Cross-validation / GridSearch | `n_jobs=-1` (joblib multiprocessing) |
| Preprocessing banyak file | ProcessPoolExecutor / joblib |
| Unduh dataset / scraping | ThreadPoolExecutor |
| Ribuan panggilan API LLM/embedding | asyncio + semaphore + retry |
| Inferensi banyak sampel | **batching** (proses sekaligus, bukan satu per satu) |
| Komputasi numerik berat | vectorize (NumPy) / GPU (PyTorch) |

```python
# PyTorch DataLoader: paralelkan loading & augmentasi data di CPU
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
# num_workers memakai multiprocessing agar GPU tak menganggur menunggu data

# Batch inference: JANGAN loop satu-satu
# ❌ for x in data: model(x)
# ✅ model(batch)    -> jauh lebih efisien (GPU suka batch besar)
```

---

## 11. JIT & Akselerasi (Numba, Cython)

Kalau ada loop Python yang **benar-benar tak bisa divectorize** (algoritma berurutan) dan jadi bottleneck:

```python
# Numba — kompilasi fungsi Python ke kode mesin via decorator (pip install numba)
from numba import njit

@njit
def hitung(arr):
    total = 0.0
    for x in arr:           # loop ini sekarang secepat C
        total += x ** 2
    return total
```
- **Numba** — paling mudah, untuk fungsi numerik. Bisa `@njit(parallel=True)`.
- **Cython** — kompilasi Python+tipe ke C, lebih banyak kontrol.
- **CuPy** — "NumPy di GPU".
- Untuk DL, biasanya cukup PyTorch (sudah CUDA).

---

## 12. Cheat Sheet & Latihan

```
Lambat?  →  PROFIL DULU (%timeit, cProfile) — jangan tebak.
│
├─ Loop atas array?         →  Vectorize (NumPy/Pandas), boolean mask, einsum
├─ Boros memori?            →  dtype lebih kecil, view bukan copy, generator/streaming
├─ Loop tak bisa divector?  →  Numba @njit
│
└─ Butuh paralel?
   ├─ I/O-bound (file/net)        →  threading (ThreadPoolExecutor) / asyncio
   ├─ Banyak API call (LLM)       →  asyncio + semaphore + retry
   ├─ CPU-bound Python murni      →  multiprocessing / joblib (n_jobs=-1)
   └─ Numerik berat (matmul)      →  sudah paralel di NumPy/GPU — cukup vectorize
```

### Latihan
1. Jalankan [benchmark.py](./benchmark.py). Bandingkan waktu loop vs vectorized untuk operasi yang sama. Berapa kali lipat lebih cepat?
2. Tulis ulang fungsi ber-loop ini secara vectorized: hitung jarak Euclidean tiap baris matriks `X` (n×d) ke satu vektor `v`. Bandingkan waktunya.
3. Demonstrasikan view vs copy: buat array, slice tanpa `.copy()`, ubah slice, dan tunjukkan array asal ikut berubah.
4. Pakai ThreadPoolExecutor untuk mengunduh/membaca 50 file/URL. Bandingkan dengan versi berurutan.
5. Pakai ProcessPoolExecutor (atau joblib) untuk komputasi CPU-bound di banyak core. Verifikasi speedup ≈ jumlah core.
6. (Lanjut) Tulis pemroses dokumen async yang memanggil API (mock) untuk 1000 item dengan batas konkurensi 20 memakai `asyncio.Semaphore`.
7. Percepat satu loop numerik dengan `@njit` Numba dan ukur perbedaannya.

⬅️ Kembali ke [Daftar Modul](../README.md) · Terkait: [Modul 00.1 · NumPy & Pandas](../00-prasyarat/01-python-data-science.md), [Modul 07 · MLOps](../07-mlops-deployment/README.md)
