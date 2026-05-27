# 00.3 · Kalkulus & Gradien

> Kalkulus menjawab pertanyaan inti ML: **"Bagaimana cara mengubah parameter agar model jadi lebih baik?"** Jawabannya selalu sama — ikuti **gradien**.

## 1. Turunan: laju perubahan

Turunan $f'(x)$ atau $\frac{df}{dx}$ = **kemiringan** fungsi di titik $x$ = seberapa cepat output berubah saat input berubah sedikit.

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

```python
import numpy as np

def f(x):
    return x ** 2          # f'(x) = 2x

# Turunan numerik (definisi):
def turunan(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)   # central difference, lebih akurat

turunan(f, 3.0)     # ~6.0  (karena 2*3 = 6)
```

### Aturan turunan dasar (hafalkan)

| Fungsi | Turunan |
|---|---|
| $c$ (konstanta) | $0$ |
| $x^n$ | $n x^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln(x)$ | $1/x$ |
| $f(x) + g(x)$ | $f' + g'$ |
| $f(x) \cdot g(x)$ | $f'g + fg'$ |

### Aturan rantai (chain rule) — JANTUNG backpropagation

Jika $y = f(g(x))$, maka:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

Contoh: $y = (3x + 1)^2$. Misal $u = 3x+1$, maka $y = u^2$.
$$\frac{dy}{dx} = \underbrace{2u}_{dy/du} \cdot \underbrace{3}_{du/dx} = 6(3x+1)$$

> **Backpropagation hanyalah aturan rantai yang diterapkan berulang** melalui banyak layer. Kalau kamu paham chain rule, kamu paham 80% backprop.

---

## 2. Turunan Parsial & Gradien

Fungsi ML punya **banyak** input (parameter). Turunan parsial $\frac{\partial f}{\partial x_i}$ = turunan terhadap satu variabel, anggap yang lain konstan.

**Gradien** $\nabla f$ = vektor berisi semua turunan parsial:

$$\nabla f = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right]$$

**Intuisi penting:** gradien menunjuk ke arah **kenaikan tercuram**. Maka untuk **meminimalkan** loss, kita bergerak ke arah **negatif** gradien.

```python
# f(x, y) = x² + y², gradien = [2x, 2y]
def grad_f(x, y):
    return np.array([2 * x, 2 * y])

grad_f(1.0, 2.0)    # [2. 4.] -> arah naik tercuram dari titik (1,2)
```

---

## 3. Gradient Descent — algoritma "belajar"

Ini adalah algoritma optimasi di balik hampir semua ML modern. Idenya:

1. Mulai dari tebakan parameter acak.
2. Hitung gradien loss terhadap parameter.
3. Geser parameter sedikit ke arah **negatif** gradien.
4. Ulangi sampai loss tidak turun lagi.

$$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$$

di mana $\alpha$ adalah **learning rate** (seberapa besar langkah).

### Implementasi dari nol: cari minimum $f(x) = x^2$

```python
import numpy as np

def f(x):        return x ** 2
def grad(x):     return 2 * x

x = 10.0              # tebakan awal
learning_rate = 0.1
history = [x]

for step in range(50):
    g = grad(x)                 # 1. hitung gradien
    x = x - learning_rate * g   # 2. langkah turun
    history.append(x)

print(f"Minimum di x = {x:.5f}")   # mendekati 0 (benar, min x² ada di 0)
```

### Visualisasi konvergensi

```python
import matplotlib.pyplot as plt

xs = np.linspace(-10, 10, 100)
plt.plot(xs, f(xs), label="f(x)=x²")
plt.scatter(history, [f(x) for x in history], c="red", s=20, label="langkah GD")
plt.legend(); plt.title("Gradient Descent menuruni kurva"); plt.show()
```

### Peran learning rate

- **Terlalu kecil** → konvergen sangat lambat.
- **Terlalu besar** → melompati minimum, bisa **divergen** (loss meledak).
- Ini **hyperparameter** terpenting yang akan sering kamu tuning.

```python
# Coba sendiri: ganti learning_rate ke 0.01 (lambat) dan 1.1 (divergen).
```

---

## 4. Varian Gradient Descent (sekilas)

| Varian | Cara hitung gradien | Trade-off |
|---|---|---|
| **Batch GD** | seluruh dataset tiap langkah | akurat tapi lambat & boros memori |
| **Stochastic GD (SGD)** | satu sampel tiap langkah | cepat, berisik, bisa lolos minimum lokal |
| **Mini-batch GD** | sebagian kecil (mis. 32 sampel) | **standar de facto** — keseimbangan terbaik |

Optimizer modern (**Adam**, **RMSprop**) menyempurnakan ide ini dengan menyesuaikan learning rate per-parameter secara adaptif. Kita pakai di Modul 04.

---

## 5. Autograd: kenapa kamu tak perlu hitung turunan manual

Library seperti **PyTorch** menghitung gradien otomatis (automatic differentiation). Kamu definisikan komputasi, framework yang menurunkannya:

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = (3 * x + 1) ** 2          # fungsi yang sama dengan contoh chain rule
y.backward()                  # hitung dy/dx otomatis
print(x.grad)                 # tensor(60.) -> sama dengan 6*(3*3+1)=60 ✓
```

> Tetap penting paham teorinya: saat debugging "gradien NaN" atau "vanishing gradient", kamu harus tahu apa yang sedang terjadi di balik layar.

---

## Latihan

1. Turunkan secara manual $f(x) = (2x + 3)^2$ dengan chain rule, lalu verifikasi dengan `turunan()` numerik dan dengan PyTorch autograd.
2. Implementasikan gradient descent untuk fungsi 2D $f(x,y) = x^2 + 3y^2$. Plot lintasan parameter menuju minimum (0,0).
3. Eksperimen learning rate: untuk $f(x)=x^2$, plot nilai $x$ tiap iterasi untuk `lr = 0.01, 0.1, 0.9, 1.01`. Jelaskan apa yang terjadi di tiap kasus.
4. (Tantangan) Implementasikan **regresi linear** sederhana dengan gradient descent dari nol — ini "Hello World" ML. Kita akan lakukan lengkap di Modul 03.

➡️ Lanjut: [00.4 · Probabilitas & Statistik](./04-probabilitas-statistik.md)
