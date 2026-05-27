# Modul 04 · Deep Learning

> Neural network adalah fungsi yang sangat fleksibel yang bisa mempelajari pola rumit dari data berlimpah (gambar, teks, suara). Modul ini membangun dari **satu neuron** sampai jaringan dalam, dengan PyTorch — framework yang dipakai industri & riset.

## Tujuan Belajar
- Memahami neuron, layer, dan **forward pass**.
- Memahami **backpropagation** (penerapan chain rule, Modul 00.3).
- Membangun & melatih neural network dengan **PyTorch**.
- Menguasai komponen training: loss, optimizer, activation, batch.
- Mengenal **CNN** dan trik mencegah overfitting (dropout, batch norm, early stopping).

## Daftar Isi
1. Dari neuron ke jaringan
2. Activation functions
3. Forward pass & loss
4. Backpropagation
5. Implementasi dari nol → [neural_network_scratch.py](./neural_network_scratch.py)
6. PyTorch: dasar tensor & autograd
7. Melatih jaringan dengan PyTorch → [pytorch_mnist.py](./pytorch_mnist.py)
8. CNN (Convolutional Neural Network)
9. Trik training & regularisasi

---

## 1. Dari Neuron ke Jaringan

Satu **neuron** = regresi logistik: hitung kombinasi linear input, lalu lewatkan ke fungsi aktivasi.

$$a = f(\mathbf{w}^T\mathbf{x} + b)$$

Susun banyak neuron jadi **layer**, tumpuk banyak layer → **neural network** (Multi-Layer Perceptron, MLP).

```
Input        Hidden Layer 1      Hidden Layer 2     Output
 x1 ─┐        (o)──┐               (o)──┐
 x2 ─┼──►(o)  (o)  ├──►(o)  (o)    (o)  ├──► ŷ
 x3 ─┘        (o)──┘               (o)──┘
        bobot W1,b1      W2,b2          W3,b3
```

**Kenapa butuh banyak layer?** Tiap layer membangun fitur lebih abstrak. Pada gambar: layer awal mendeteksi tepi → bentuk → objek. Inilah **representation learning** — model menemukan fiturnya sendiri (beda dari classical ML yang fiturnya kita buat manual).

> **Universal Approximation Theorem:** jaringan dengan cukup neuron bisa mengaproksimasi fungsi kontinu apa pun. Kekuatannya dari sini.

---

## 2. Activation Functions

Tanpa aktivasi non-linear, menumpuk layer linear tetap = satu fungsi linear. Aktivasi memberi **non-linearitas**.

| Fungsi | Rumus | Catatan |
|---|---|---|
| **ReLU** | max(0, x) | default untuk hidden layer — cepat, hindari vanishing gradient |
| **Sigmoid** | 1/(1+e⁻ˣ) | output [0,1], untuk probabilitas biner; rawan vanishing |
| **Tanh** | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | output [−1,1] |
| **Softmax** | eˣⁱ/Σeˣ | output layer klasifikasi multi-kelas (jadi distribusi probabilitas) |
| **GELU/SiLU** | varian halus ReLU | dipakai di Transformer modern |

```python
import numpy as np
def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))   # -max untuk stabilitas numerik
    return e / e.sum(axis=-1, keepdims=True)
```

---

## 3. Forward Pass & Loss

**Forward pass** = alirkan input lewat tiap layer untuk dapat prediksi.

```
h1 = relu(X  @ W1 + b1)
h2 = relu(h1 @ W2 + b2)
ŷ  = softmax(h2 @ W3 + b3)
```

**Loss** mengukur kesalahan prediksi:
- Regresi → **MSE**.
- Klasifikasi → **Cross-Entropy** (Modul 00.4).

---

## 4. Backpropagation — bagaimana jaringan belajar

Tujuan: cari gradien loss terhadap **setiap** bobot, lalu update dengan gradient descent. Backprop menghitung gradien ini secara efisien dengan **chain rule** dari output mundur ke input.

```
Forward  ──────────────────────────────►
 X → [Layer1] → [Layer2] → [Layer3] → Loss
       ◄────────────────────────────── Backward
       gradien mengalir mundur (chain rule)
```

Tiga langkah inti training:
1. **Forward** → hitung prediksi & loss.
2. **Backward** → hitung gradien tiap parameter (backprop).
3. **Update** → `param -= lr * grad` (optimizer).

> Kamu **tidak** akan menulis backprop manual di kerja nyata — PyTorch melakukannya otomatis (autograd). Tapi kita implementasikan sekali dari nol agar tidak jadi "black box".

---

## 5. Implementasi dari Nol

📂 Lihat & jalankan [neural_network_scratch.py](./neural_network_scratch.py): MLP 2-layer untuk klasifikasi, lengkap dengan forward, backprop manual, dan training loop — **hanya NumPy**. Ini momen "aha!" untuk memahami deep learning.

---

## 6. PyTorch: Tensor & Autograd

**Tensor** = array NumPy yang bisa jalan di GPU & melacak gradien.

```python
import torch

# Tensor mirip NumPy
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x.shape, x @ x, x.mean()

# Autograd: lacak operasi untuk hitung gradien otomatis
w = torch.tensor(2.0, requires_grad=True)
loss = (w - 5) ** 2
loss.backward()           # hitung dL/dw
print(w.grad)             # tensor(-6.) -> 2*(2-5)

# Pindah ke GPU jika ada
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
```

---

## 7. Melatih Jaringan dengan PyTorch

Pola standar (`nn.Module` + loss + optimizer + training loop):

```python
import torch
import torch.nn as nn

# 1. Definisikan model
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=20, hidden=64, out_dim=2)

# 2. Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. Training loop
for epoch in range(20):
    model.train()
    for xb, yb in train_loader:          # mini-batch
        optimizer.zero_grad()            # reset gradien
        out = model(xb)                  # forward
        loss = criterion(out, yb)        # hitung loss
        loss.backward()                  # backprop (autograd)
        optimizer.step()                 # update bobot

    # Validasi
    model.eval()
    with torch.no_grad():
        ...  # hitung akurasi di val_loader
```

📂 Contoh lengkap pada dataset gambar (MNIST/digits): [pytorch_mnist.py](./pytorch_mnist.py).

### Komponen yang perlu dipahami
- **DataLoader** — membagi data jadi mini-batch & shuffling.
- **optimizer.zero_grad()** — WAJIB; gradien terakumulasi kalau tidak di-reset.
- **model.train() / model.eval()** — mengubah perilaku Dropout & BatchNorm.
- **torch.no_grad()** — matikan pelacakan gradien saat inferensi (hemat memori).
- **Optimizer**: **Adam** (default andal), SGD+momentum (klasik, sering untuk CNN besar).

---

## 8. Convolutional Neural Network (CNN)

Untuk **gambar** (dibahas lebih dalam di Modul 06). MLP boros parameter untuk gambar; CNN memakai **konvolusi** — filter kecil yang digeser ke seluruh gambar untuk mendeteksi pola lokal (tepi, tekstur) dengan **berbagi bobot**.

```python
import torch.nn as nn
cnn = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
```

Komponen: **Conv2d** (deteksi fitur), **MaxPool** (downsampling), **Flatten** → fully connected → output.

---

## 9. Trik Training & Regularisasi

| Teknik | Fungsi |
|---|---|
| **Normalisasi input** | konvergensi lebih cepat & stabil |
| **Dropout** | matikan neuron acak saat training → cegah overfitting |
| **Batch Normalization** | normalisasi aktivasi antar layer → training stabil & cepat |
| **Early Stopping** | berhenti saat loss validasi mulai naik |
| **Learning rate scheduling** | turunkan lr seiring waktu untuk konvergensi halus |
| **Weight decay (L2)** | regularisasi bobot |
| **Data augmentation** | perbanyak variasi data (Modul 06) |
| **Gradient clipping** | cegah gradien meledak (penting di RNN/Transformer) |

### Masalah klasik
- **Vanishing gradient** — gradien mengecil di layer dalam → pakai ReLU, BatchNorm, residual connection.
- **Exploding gradient** — gradien membesar → gradient clipping.
- **Overfitting** — train bagus, val jelek → dropout, augmentasi, lebih banyak data, model lebih kecil.

---

## Ringkasan
1. Neural net = layer linear + aktivasi non-linear yang ditumpuk.
2. Belajar lewat siklus **forward → loss → backward → update**.
3. **Backprop = chain rule**; autograd PyTorch melakukannya otomatis.
4. **CNN** untuk gambar; pakai trik regularisasi untuk generalisasi.

## Latihan
1. Jalankan & pahami [neural_network_scratch.py](./neural_network_scratch.py). Ubah jumlah neuron hidden — apa efeknya?
2. Jalankan [pytorch_mnist.py](./pytorch_mnist.py). Capai akurasi >97%. Plot kurva loss train vs val.
3. Tambahkan/hapus Dropout dan amati efeknya pada overfitting (gap train-val).
4. Ganti MLP dengan CNN pada dataset gambar. Bandingkan akurasi & jumlah parameter.
5. Eksperimen optimizer (SGD vs Adam) dan learning rate. Mana yang konvergen lebih cepat?

➡️ Lanjut: [Modul 05 · NLP & LLM](../05-nlp-llm/README.md) atau [Modul 06 · Computer Vision](../06-computer-vision/README.md)
