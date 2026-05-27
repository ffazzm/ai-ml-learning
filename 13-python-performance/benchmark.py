"""
Benchmark performa: loop Python vs NumPy vectorization vs paralel.

Menunjukkan dengan angka NYATA seberapa besar bedanya. Jalankan dan lihat
sendiri kenapa AI Engineer "berpikir vectorized".

Jalankan:  python benchmark.py
"""
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np


def timer(label, fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    elapsed = time.perf_counter() - start
    print(f"  {label:<28} {elapsed:8.4f}s")
    return elapsed, result


# ============================================================
# 1. VECTORIZATION vs FOR LOOP (operasi numerik elemen-per-elemen)
# ============================================================
def demo_vectorization():
    print("\n[1] Vectorization vs Loop  (y = x*2 + 1, 10 juta elemen)")
    x = np.arange(10_000_000, dtype=np.float64)

    def pakai_loop(arr):
        out = np.empty_like(arr)
        for i in range(len(arr)):
            out[i] = arr[i] * 2 + 1
        return out

    def pakai_vectorized(arr):
        return arr * 2 + 1

    t_loop, _ = timer("Loop Python", pakai_loop, x)
    t_vec, _ = timer("NumPy vectorized", pakai_vectorized, x)
    print(f"  -> Vectorized {t_loop / t_vec:.0f}x lebih cepat\n")


# ============================================================
# 2. Contoh praktis: jarak Euclidean tiap baris ke satu vektor
# ============================================================
def demo_jarak():
    print("[2] Jarak Euclidean 100k baris (d=50) ke satu vektor")
    X = np.random.randn(100_000, 50)
    v = np.random.randn(50)

    def pakai_loop(X, v):
        out = np.empty(len(X))
        for i in range(len(X)):
            out[i] = math.sqrt(np.sum((X[i] - v) ** 2))
        return out

    def pakai_vectorized(X, v):
        return np.sqrt(((X - v) ** 2).sum(axis=1))   # broadcasting + agregasi

    t_loop, _ = timer("Loop Python", pakai_loop, X, v)
    t_vec, _ = timer("NumPy vectorized", pakai_vectorized, X, v)
    print(f"  -> Vectorized {t_loop / t_vec:.0f}x lebih cepat\n")


# ============================================================
# 3. CPU-bound: serial vs multiprocessing vs threading
#    (menunjukkan GIL: threading TIDAK menolong tugas CPU-bound)
# ============================================================
def kerja_cpu(n):
    return sum(math.sqrt(i) for i in range(n))   # CPU-bound Python murni


def demo_parallel():
    print("[3] CPU-bound (8 tugas berat): serial vs thread vs process")
    tugas = [3_000_000] * 8

    def serial(tugas):
        return [kerja_cpu(n) for n in tugas]

    def pakai_thread(tugas):
        with ThreadPoolExecutor(max_workers=4) as ex:
            return list(ex.map(kerja_cpu, tugas))

    def pakai_process(tugas):
        with ProcessPoolExecutor(max_workers=4) as ex:
            return list(ex.map(kerja_cpu, tugas))

    timer("Serial", serial, tugas)
    timer("Threading (4)", pakai_thread, tugas)      # ~ sama dgn serial krn GIL!
    timer("Multiprocessing (4)", pakai_process, tugas)  # lebih cepat (paralel sejati)
    print("  -> Threading TAK menolong CPU-bound (GIL); multiprocessing menolong.\n")


if __name__ == "__main__":
    demo_vectorization()
    demo_jarak()
    demo_parallel()
