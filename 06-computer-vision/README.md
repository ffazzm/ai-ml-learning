# Modul 06 · Computer Vision

> Mengajari mesin "melihat". Dari piksel mentah sampai mendeteksi objek dan memahami isi gambar. Modul ini memperdalam CNN (Modul 04) dan teknik praktis yang dipakai di produksi: **transfer learning** & **data augmentation**.

## Tujuan Belajar
- Memahami representasi gambar sebagai tensor.
- Menguasai komponen & arsitektur **CNN**.
- Memakai **transfer learning** (cara #1 menyelesaikan masalah CV nyata).
- Menerapkan **data augmentation**.
- Mengenal tugas: klasifikasi, **deteksi objek**, **segmentasi**.

## Daftar Isi
1. Gambar sebagai data
2. Konvolusi & komponen CNN
3. Arsitektur CNN penting
4. Transfer Learning ⭐
5. Data Augmentation
6. Tugas CV: deteksi & segmentasi
7. Vision Transformer & tren modern

---

## 1. Gambar sebagai Data

Gambar = tensor angka (intensitas piksel 0–255).

- **Grayscale:** matriks `(tinggi, lebar)`.
- **Berwarna:** tensor `(tinggi, lebar, 3)` — channel R, G, B.
- **Batch gambar:** `(N, C, H, W)` di PyTorch (channel-first).

```python
import torch
from torchvision import transforms
from PIL import Image

img = Image.open("foto.jpg")
to_tensor = transforms.ToTensor()      # PIL -> tensor [0,1], shape (3, H, W)
x = to_tensor(img)
x.shape, x.min(), x.max()
```

Normalisasi (kurangi mean, bagi std) penting agar training stabil — biasanya pakai statistik ImageNet untuk model pretrained.

---

## 2. Konvolusi & Komponen CNN

Kenapa bukan MLP biasa? Gambar 224×224×3 = 150.528 input → MLP butuh parameter raksasa & buang struktur spasial. **CNN** mengeksploitasi struktur lokal & berbagi bobot.

### Operasi konvolusi
Sebuah **filter/kernel** kecil (mis. 3×3) digeser ke seluruh gambar, menghitung dot product di tiap posisi → **feature map** yang menonjolkan pola (tepi, tekstur). Filter dipelajari otomatis lewat backprop.

```
Filter belajar mendeteksi pola; layer awal -> tepi/warna,
layer dalam -> tekstur -> bagian objek -> objek utuh (hierarki fitur).
```

### Komponen
| Layer | Fungsi |
|---|---|
| **Conv2d** | ekstraksi fitur via filter (params: kernel_size, stride, padding) |
| **Activation (ReLU)** | non-linearitas |
| **Pooling (MaxPool)** | downsampling → ringkas & invarian translasi |
| **Batch Norm** | stabilkan & percepat training |
| **Flatten + Linear** | klasifikasi akhir dari fitur |

```python
import torch.nn as nn
cnn = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Linear(64, 10),
)
```

---

## 3. Arsitektur CNN Penting (kenali namanya)

| Arsitektur | Kontribusi |
|---|---|
| **LeNet-5** (1998) | CNN pertama yang sukses (digit) |
| **AlexNet** (2012) | memicu revolusi deep learning (menang ImageNet) |
| **VGG** (2014) | sederhana & dalam, filter 3×3 ditumpuk |
| **ResNet** (2015) | **residual/skip connection** → bisa sangat dalam (50–152 layer); paling sering jadi backbone |
| **EfficientNet** | scaling seimbang → akurasi tinggi, efisien |

> **ResNet** adalah default andal untuk transfer learning. Skip connection mengatasi vanishing gradient sehingga jaringan ratusan layer bisa dilatih.

---

## 4. Transfer Learning ⭐ (cara nyata menyelesaikan CV)

Melatih CNN dari nol butuh jutaan gambar & GPU mahal. **Transfer learning:** ambil model yang sudah dilatih di ImageNet (jutaan gambar), lalu sesuaikan untuk tugasmu dengan data jauh lebih sedikit. **Ini yang dipakai 90% kasus nyata.**

Dua strategi:
- **Feature extraction** — bekukan backbone, latih hanya layer klasifikasi baru (data sedikit).
- **Fine-tuning** — latih juga sebagian/seluruh backbone dengan lr kecil (data lebih banyak).

```python
import torch.nn as nn
from torchvision import models

# 1. Ambil ResNet18 pretrained
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. Bekukan semua bobot backbone
for p in model.parameters():
    p.requires_grad = False

# 3. Ganti layer akhir sesuai jumlah kelas kita (mis. 5 kelas)
model.fc = nn.Linear(model.fc.in_features, 5)   # hanya ini yang dilatih

# 4. Latih seperti biasa (optimizer hanya menerima model.fc.parameters())
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

> 💡 Dengan transfer learning, kamu bisa dapat akurasi tinggi dari **ratusan** gambar saja, dalam menit di GPU gratis (Colab/Kaggle).

---

## 5. Data Augmentation

Perbanyak variasi data dengan transformasi acak → model lebih general & tahan overfitting. Hanya pada **data training**.

```python
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # statistik ImageNet
                         std=[0.229, 0.224, 0.225]),
])

# Validasi/test: TANPA augmentasi acak, hanya resize + normalize
val_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

⚠️ Pilih augmentasi yang masuk akal: flip horizontal OK untuk kucing, **tidak** untuk digit/teks (angka 6 ≠ 9 terbalik).

---

## 6. Tugas Computer Vision

| Tugas | Output | Model populer |
|---|---|---|
| **Klasifikasi** | label gambar | ResNet, EfficientNet |
| **Deteksi objek** | bounding box + label | YOLO, Faster R-CNN |
| **Segmentasi semantik** | label per piksel | U-Net, DeepLab |
| **Segmentasi instance** | objek terpisah per piksel | Mask R-CNN, SAM |
| **Pose estimation** | titik sendi tubuh | OpenPose |

```python
# Deteksi objek siap pakai dengan torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
detector = fasterrcnn_resnet50_fpn(weights="DEFAULT").eval()
# output: dict berisi 'boxes', 'labels', 'scores'

# YOLO (ultralytics) — populer untuk real-time:
# from ultralytics import YOLO
# model = YOLO("yolov8n.pt"); model("foto.jpg")
```

---

## 7. Vision Transformer & Tren Modern

- **ViT (Vision Transformer)** — terapkan Transformer (Modul 05) ke gambar dengan memecahnya jadi patch. Mengungguli CNN pada dataset besar.
- **CLIP** — hubungkan gambar & teks dalam satu ruang embedding → klasifikasi zero-shot ("cari gambar yang cocok teks ini").
- **SAM (Segment Anything)** — segmentasi apa pun tanpa training khusus.
- **Diffusion models** (Stable Diffusion, DALL·E) — generasi gambar dari teks.
- **Multimodal LLM** — model yang melihat & membaca sekaligus (mis. menjawab pertanyaan tentang gambar).

---

## Ringkasan
1. Gambar = tensor; **CNN** mengeksploitasi struktur spasial via konvolusi.
2. Hierarki fitur: tepi → bentuk → objek.
3. **Transfer learning** = cara praktis #1; jarang latih dari nol.
4. **Augmentation** memerangi overfitting.
5. Selain klasifikasi: deteksi, segmentasi, dan model multimodal modern.

## Latihan
1. Latih CNN sederhana dari nol pada CIFAR-10 atau Fashion-MNIST. Catat akurasi.
2. Selesaikan tugas yang sama dengan **transfer learning** (ResNet18). Bandingkan akurasi & waktu — rasakan kekuatannya.
3. Tambahkan data augmentation. Apakah gap train-val mengecil?
4. Pakai detektor objek pretrained pada fotomu sendiri. Visualisasikan bounding box.
5. Coba CLIP untuk klasifikasi zero-shot tanpa training sama sekali.

➡️ Lanjut: [Modul 07 · MLOps & Deployment](../07-mlops-deployment/README.md)
