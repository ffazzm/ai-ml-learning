# 00.4 · Probabilitas & Statistik

> ML adalah **belajar dari data yang tidak pasti**. Probabilitas memberi bahasa untuk ketidakpastian; statistik memberi alat untuk menyimpulkan dari sampel.

## 1. Statistik Deskriptif

```python
import numpy as np
data = np.array([2, 4, 4, 4, 5, 5, 7, 9])

np.mean(data)     # 5.0   -> rata-rata (sensitif outlier)
np.median(data)   # 4.5   -> nilai tengah (tahan outlier)
np.std(data)      # 2.0   -> sebaran (standar deviasi)
np.var(data)      # 4.0   -> variansi = std²
np.percentile(data, 25)   # kuartil bawah
```

- **Mean vs Median:** kalau data punya outlier (mis. gaji CEO di data karyawan), **median** lebih mewakili. Ini keputusan penting di feature engineering (Modul 02).
- **Variansi/Std:** mengukur seberapa "tersebar" data. Inti dari standardisasi fitur.

---

## 2. Probabilitas Dasar

- $P(A)$ ∈ [0, 1]: probabilitas kejadian A.
- $P(A^c) = 1 - P(A)$: komplemen.
- **Independen:** $P(A \cap B) = P(A) \cdot P(B)$.
- **Conditional:** $P(A|B)$ = probabilitas A **jika** B sudah terjadi.

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

> Conditional probability adalah inti klasifikasi: model memprediksi $P(\text{label} \mid \text{fitur})$.

---

## 3. Teorema Bayes — fondasi inferensi

$$P(A|B) = \frac{P(B|A) \, P(A)}{P(B)}$$

Membalik arah kondisional: dari $P(\text{data}|\text{hipotesis})$ jadi $P(\text{hipotesis}|\text{data})$.

**Contoh klasik — tes penyakit:**
- Penyakit langka: $P(\text{sakit}) = 0.001$.
- Tes akurat 99%: $P(\text{positif}|\text{sakit}) = 0.99$, $P(\text{positif}|\text{sehat}) = 0.01$.
- Pertanyaan: kalau tes **positif**, berapa peluang benar-benar sakit?

```python
p_sakit = 0.001
p_pos_jika_sakit = 0.99
p_pos_jika_sehat = 0.01

# P(positif) total:
p_pos = p_pos_jika_sakit * p_sakit + p_pos_jika_sehat * (1 - p_sakit)
# Bayes:
p_sakit_jika_pos = (p_pos_jika_sakit * p_sakit) / p_pos
print(f"{p_sakit_jika_pos:.3f}")   # ~0.090 -> cuma 9%!
```

**Pelajaran:** meski tes 99% akurat, karena penyakitnya langka, positif palsu mendominasi. Intuisi ini penting saat menafsirkan **precision** model pada kelas tidak seimbang (Modul 01). Algoritma **Naive Bayes** (Modul 03) dibangun langsung dari teorema ini.

---

## 4. Distribusi Probabilitas

### Distribusi Normal (Gaussian) — yang paling penting

Berbentuk lonceng, ditentukan oleh mean $\mu$ dan std $\sigma$. Banyak fenomena alam & error model mendekati normal.

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

```python
import numpy as np
import matplotlib.pyplot as plt

samples = np.random.normal(loc=0, scale=1, size=10000)  # mean=0, std=1
plt.hist(samples, bins=50, density=True)
plt.title("Distribusi Normal Standar"); plt.show()
```

**Aturan 68-95-99.7:** ~68% data dalam ±1σ, ~95% dalam ±2σ, ~99.7% dalam ±3σ. Dipakai untuk deteksi outlier.

### Distribusi lain yang sering muncul

| Distribusi | Untuk | Contoh |
|---|---|---|
| **Bernoulli** | satu kejadian biner | lempar koin, klik/tidak |
| **Binomial** | jumlah sukses dari n percobaan | 7 dari 10 prediksi benar |
| **Uniform** | semua nilai sama peluang | inisialisasi bobot acak |
| **Poisson** | jumlah kejadian per interval | banyak request per detik |

---

## 5. Konsep Statistik untuk Evaluasi Model

### Sampel vs Populasi
Kita melatih model dari **sampel** (data train) dan berharap ia bekerja di **populasi** (dunia nyata). Inti dari generalisasi — kenapa kita butuh data **test** terpisah (Modul 01).

### Korelasi
Mengukur hubungan linear antara dua variabel, antara -1 dan +1.

```python
import numpy as np
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 6])
np.corrcoef(x, y)[0, 1]    # koefisien korelasi Pearson
```

> ⚠️ **Korelasi ≠ kausalitas.** Penjualan es krim berkorelasi dengan kasus tenggelam — keduanya disebabkan musim panas, bukan satu menyebabkan yang lain.

### Hukum Bilangan Besar & Central Limit Theorem
- **LLN:** makin banyak sampel, rata-rata sampel makin mendekati rata-rata sebenarnya. (Kenapa lebih banyak data → model lebih baik.)
- **CLT:** rata-rata dari banyak sampel cenderung berdistribusi normal, apa pun distribusi aslinya. (Dasar banyak uji statistik.)

---

## 6. Entropi & Cross-Entropy (penghubung ke loss function)

**Entropi** mengukur ketidakpastian/"kekagetan" rata-rata sebuah distribusi:

$$H(p) = -\sum_i p_i \log p_i$$

**Cross-entropy** mengukur seberapa berbeda prediksi $q$ dari kebenaran $p$ — ini adalah **loss function** standar untuk klasifikasi:

$$H(p, q) = -\sum_i p_i \log q_i$$

```python
import numpy as np

def cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)   # hindari log(0)
    return -np.sum(y_true * np.log(y_pred))

# Prediksi bagus (yakin & benar) -> loss kecil
cross_entropy(np.array([1, 0, 0]), np.array([0.9, 0.05, 0.05]))   # ~0.105
# Prediksi buruk (yakin tapi salah) -> loss besar
cross_entropy(np.array([1, 0, 0]), np.array([0.1, 0.8, 0.1]))     # ~2.30
```

> Setiap kali model klasifikasi "belajar", ia meminimalkan cross-entropy lewat gradient descent. Konsep ini menyatukan probabilitas + kalkulus + ML.

---

## Latihan

1. Simulasikan 10.000 lemparan dua dadu. Plot histogram jumlahnya. Apakah mendekati normal? (Petunjuk: CLT.)
2. Hitung ulang contoh Bayes tes penyakit, tapi untuk penyakit umum ($P=0.3$). Bandingkan hasilnya — kenapa berubah drastis?
3. Buat dua array, satu berkorelasi kuat dan satu acak. Hitung & visualisasikan korelasinya dengan scatter plot.
4. Implementasikan fungsi entropi. Hitung entropi koin adil (p=0.5) vs koin berat sebelah (p=0.9). Mana yang lebih "tidak pasti"?

---

🎉 **Selesai Modul 00!** Kamu sekarang punya fondasi Python + matematika. Saatnya membangun di atasnya.

➡️ Lanjut: [Modul 01 · Fondasi ML](../01-fondasi-ml/README.md)
