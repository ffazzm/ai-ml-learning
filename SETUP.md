# 🛠️ Setup Lingkungan Belajar

## 1. Install Python

Gunakan **Python 3.10+**. Cek versi:

```bash
python --version
```

Kalau belum ada, install dari [python.org](https://www.python.org/downloads/) atau pakai [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (direkomendasikan untuk data science karena mempermudah manajemen environment).

## 2. Buat Virtual Environment

Selalu pisahkan dependency per project agar tidak bentrok.

### Opsi A — `venv` (bawaan Python)

```bash
python -m venv .venv

# Aktifkan:
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows (PowerShell)
```

### Opsi B — `conda`

```bash
conda create -n aiml python=3.11 -y
conda activate aiml
```

## 3. Install Dependency

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Jalankan Jupyter / Lab

Banyak materi cocok dieksplorasi interaktif:

```bash
jupyter lab
# atau
jupyter notebook
```

VS Code juga punya dukungan notebook bagus (extension "Jupyter").

## 5. Verifikasi Instalasi

Jalankan skrip cek di bawah:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch

print("NumPy        :", np.__version__)
print("Pandas       :", pd.__version__)
print("scikit-learn :", sklearn.__version__)
print("PyTorch      :", torch.__version__)
print("CUDA tersedia:", torch.cuda.is_available())
```

Kalau semua tercetak tanpa error, kamu siap. `CUDA tersedia: False` itu wajar kalau kamu tidak punya GPU NVIDIA — semua materi awal jalan baik di CPU.

## Catatan GPU

- Deep learning besar **lebih cepat** dengan GPU, tapi tidak wajib untuk belajar.
- Gratisan: **Google Colab** (`colab.research.google.com`) dan **Kaggle Notebooks** menyediakan GPU gratis.
- Untuk install PyTorch dengan CUDA spesifik, ikuti [pytorch.org/get-started](https://pytorch.org/get-started/locally/).
