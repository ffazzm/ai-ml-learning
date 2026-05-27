# 🧠 AI/ML Engineering — Kurikulum Lengkap

> Materi belajar AI/ML dari **teori sampai coding**, disusun untuk engineer dari level pemula sampai mahir yang ingin menjadi **AI/ML Engineer** dengan fondasi yang kuat.

Kurikulum ini tidak hanya mengajarkan "cara memanggil library", tapi membangun **intuisi + matematika + implementasi dari nol + praktik produksi**. Setiap modul punya: penjelasan teori, kode yang bisa dijalankan, dan latihan.

---

## 🎯 Untuk siapa materi ini?

- **Software engineer** yang mau pindah/menambah skill ke AI/ML.
- **Fresh graduate / mahasiswa** yang mau fondasi kuat sebelum kerja.
- **Data analyst** yang mau naik level ke modeling & deployment.
- Siapa saja yang ingin paham AI **sampai ke akarnya**, bukan sekadar copy-paste.

**Prasyarat minimum:** bisa logika pemrograman dasar (variabel, loop, fungsi). Matematika SMA sudah cukup untuk memulai — sisanya kita bangun di Modul 00.

---

## 🗺️ Peta Belajar (Learning Path)

```
        JALUR INTI (urutkan)                    TOPIK TERAPAN (paralel, sesuai kebutuhan)
  ┌─────────────────────────┐
  │  00 · Prasyarat          │            ┌──────────────────────────────────────────┐
  │  Python + Matematika     │            │  12 · SQL untuk ML                         │
  └────────────┬────────────┘  ··········▶│      (bisa mulai sejak Modul 02)           │
               │                          ├──────────────────────────────────────────┤
  ┌────────────▼────────────┐            │  13 · Performa & Concurrency Python        │
  │  01 · Fondasi ML         │            │      (vectorization, GIL, paralel)         │
  └────────────┬────────────┘            └──────────────────────────────────────────┘
               │
  ┌────────────▼────────────┐
  │  02 · Data & Features    │
  └────────────┬────────────┘
               │
  ┌────────────▼────────────┐            ┌──────────────────────────────────────────┐
  │  03 · Classical ML       │  ·········▶│  09 · Evaluasi & Interpretasi             │
  └────────────┬────────────┘            │      (error analysis, SHAP, debugging)     │
               │                          ├──────────────────────────────────────────┤
  ┌────────────▼────────────┐            │  10 · Imbalanced Data (fraud/churn/medis)  │
  │  04 · Deep Learning      │  ·········▶│  11 · Time Series Forecasting              │
  └──────┬──────────┬────────┘            │  (mulai kapan saja setelah Modul 03)       │
         │          │                      └──────────────────────────────────────────┘
  ┌──────▼──────┐ ┌─▼───────────────┐          ▲
  │ 05 · NLP &  │ │ 06 · Computer   │          ┊  (terapkan ke setiap modul modeling)
  │     LLM     │ │      Vision     │··········┘
  └──────┬──────┘ └─┬───────────────┘
         │          │
  ┌──────▼──────────▼────────┐
  │  07 · MLOps & Deployment  │
  └────────────┬─────────────┘
               │
  ┌────────────▼────────────┐
  │  08 · Projects & Karier  │
  └─────────────────────────┘
```

**Cara membaca:** ikuti **jalur inti 00→08 berurutan** (05/06 pilih sesuai minat). Modul **09–13** adalah topik terapan yang **dikerjakan paralel** begitu prasyaratnya terpenuhi — bukan setelah selesai semua.

---

## 📚 Daftar Modul

| # | Modul | Isi | Estimasi |
|---|-------|-----|----------|
| 00 | [Prasyarat](./00-prasyarat/README.md) | Dasar Python dari nol, NumPy, Pandas, kalkulus, aljabar linear, probabilitas, statistik | 3–4 minggu |
| 01 | [Fondasi ML](./01-fondasi-ml/README.md) | Apa itu ML, jenis-jenis, bias-variance, overfitting, evaluasi, workflow | 2 minggu |
| 02 | [Data & Feature Engineering](./02-data-feature-engineering/README.md) | EDA, cleaning, encoding, scaling, feature selection, pipeline | 2 minggu |
| 03 | [Classical ML](./03-classical-ml/README.md) | Regresi, klasifikasi, tree, ensemble, clustering, dari nol + scikit-learn | 4 minggu |
| 04 | [Deep Learning](./04-deep-learning/README.md) | Neural network dari nol, backprop, PyTorch, CNN, training tricks | 4 minggu |
| 05 | [NLP & LLM / GenAI](./05-nlp-llm/README.md) | Text processing, embedding, Transformer, LLM, RAG, fine-tuning, prompt engineering | 4 minggu |
| 06 | [Computer Vision](./06-computer-vision/README.md) | Image basics, CNN arsitektur, transfer learning, deteksi & segmentasi | 3 minggu |
| 07 | [MLOps & Deployment](./07-mlops-deployment/README.md) | Serving API, Docker, tracking, monitoring, drift, CI/CD | 3 minggu |
| 09 | [Evaluasi & Interpretasi Model](./09-evaluasi-interpretasi/README.md) | Error analysis, learning curve, calibration, SHAP/LIME, Grad-CAM, debugging "kenapa akurasinya begini" | 2 minggu |
| 10 | [Imbalanced Data](./10-imbalanced-data/README.md) | Metrik untuk data timpang, class weight, SMOTE & varian, threshold tuning, anomaly detection | 1 minggu |
| 11 | [Time Series Forecasting](./11-time-series/README.md) | Komponen deret, fitur lag/rolling, validasi kronologis, ARIMA, XGBoost, Prophet, deep learning | 2 minggu |
| 12 | [SQL untuk AI/ML Engineer](./12-sql-untuk-ml/README.md) | Query, agregasi, JOIN, window functions, feature engineering di SQL, SQL→Pandas | 1–2 minggu |
| 13 | [Performa & Concurrency Python](./13-python-performance/README.md) | Vectorization vs loop, memory/dtype, profiling, GIL, threading/multiprocessing/asyncio, pola paralel ML | 1–2 minggu |
| 08 | [Projects & Roadmap Karier](./08-projects-karier/README.md) | Portfolio project end-to-end, persiapan interview, roadmap | berkelanjutan |

> 📌 Modul **09–13** adalah topik terapan yang bisa dikerjakan paralel: **09** (diagnosis & menjelaskan model) melengkapi setiap modul modeling; **10** (data timpang) krusial untuk fraud/churn/medis; **11** (time series) untuk masalah berbasis waktu; **12** (SQL) skill harian menarik & menyiapkan data; **13** (performa & concurrency) untuk menulis kode yang cepat & scalable — penting saat naik ke level engineering/produksi.

---

## 🧭 Cara memakai materi ini

1. **Jangan loncat-loncat di awal.** Modul 00–04 adalah fondasi. Sisanya (05/06) bisa dipilih sesuai minat.
2. **Ketik ulang kodenya**, jangan cuma baca. Pemahaman datang dari mengetik & debug.
3. **Kerjakan latihan** di akhir tiap modul sebelum lanjut.
4. **Bangun project** dari Modul 08 secara paralel — teori tanpa praktik cepat hilang.

### Filosofi: "Implement from scratch, then use the library"

Untuk setiap algoritma penting, kita **implementasi manual dulu** (pakai NumPy) untuk paham cara kerjanya, **baru** pakai library production (scikit-learn / PyTorch). Ini yang membedakan AI Engineer yang kuat vs yang cuma "tukang import".

---

## 🛠️ Setup Lingkungan

Lihat [SETUP.md](./SETUP.md) untuk instalasi Python, virtual environment, dan semua dependency.

```bash
# Ringkas:
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

## 📖 Sumber Lanjutan (gratis & berkualitas)

- **3Blue1Brown** (YouTube) — intuisi visual aljabar linear, kalkulus, neural network.
- **StatQuest** (YouTube) — statistik & ML dijelaskan sangat sederhana.
- **fast.ai** — kursus deep learning praktis.
- **CS229 (Stanford)** — ML teori. **CS231n** — CV. **CS224n** — NLP.
- **"Hands-On Machine Learning"** (Géron) — buku referensi terbaik untuk praktisi.
- **"The Hundred-Page ML Book"** (Burkov) — ringkas & padat.
- **Dive into Deep Learning (d2l.ai)** — buku gratis dengan kode.

---

*Materi ini hidup — perbaiki, tambah catatan, dan sesuaikan dengan kecepatan belajarmu. Selamat belajar! 🚀*

---

> 🤖 **Catatan:** Kurikulum ini dibuat dengan bantuan **Claude** (Anthropic) via [Claude Code](https://claude.com/claude-code). Gunakan sebagai panduan belajar, dan verifikasi konsep penting dengan sumber resmi & praktik langsung.
