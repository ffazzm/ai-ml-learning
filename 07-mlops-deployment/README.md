# Modul 07 · MLOps & Deployment

> Model di notebook = nol nilai bisnis. Yang membedakan **AI/ML Engineer** dari sekadar "yang bisa training model" adalah kemampuan membawa model ke **produksi** dan menjaganya tetap bekerja. Ini skill yang membuatmu dipekerjakan.

## Tujuan Belajar
- Menyimpan & memuat model dengan benar.
- Menyajikan model lewat **REST API** (FastAPI).
- Mengemas dengan **Docker**.
- Melacak eksperimen (**MLflow**).
- Memantau model di produksi & memahami **data/model drift**.
- Memahami alur **CI/CD** untuk ML.

## Daftar Isi
1. Siklus hidup MLOps
2. Menyimpan & memuat model
3. Menyajikan model: REST API → [serve_api.py](./serve_api.py)
4. Kontainerisasi dengan Docker
5. Experiment tracking (MLflow)
6. Monitoring & drift
7. CI/CD & otomatisasi
8. Skala & optimasi inferensi

---

## 1. Siklus Hidup MLOps

MLOps = DevOps untuk ML, dengan tantangan ekstra: **kode + data + model** semuanya berubah.

```
   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │  Data    │──►│  Train   │──►│ Evaluate │──►│ Register │
   └──────────┘   └──────────┘   └──────────┘   └────┬─────┘
        ▲                                            │
        │                                            ▼
   ┌────┴─────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ Re-train │◄──│  Monitor │◄──│  Serve   │◄──│  Deploy  │
   └──────────┘   └──────────┘   └──────────┘   └──────────┘
        ▲              │ drift terdeteksi → picu re-train
        └──────────────┘
```

Perbedaan dari software biasa: performa model **memburuk seiring waktu** karena dunia berubah (data drift). Maka monitoring & re-training adalah bagian inti, bukan tambahan.

---

## 2. Menyimpan & Memuat Model

### scikit-learn — `joblib`
```python
import joblib

# Simpan SELURUH pipeline (preprocessing + model), bukan hanya model!
joblib.dump(pipeline, "model.joblib")

# Muat di server
pipeline = joblib.load("model.joblib")
pipeline.predict(X_baru)
```

### PyTorch — simpan `state_dict`
```python
import torch
torch.save(model.state_dict(), "model.pt")        # cara yang dianjurkan

# Memuat: buat arsitektur, lalu load bobot
model = MyModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()
```

> ⚠️ **Jebakan umum:** menyimpan model tapi lupa preprocessing/scaler. Input produksi harus melewati transformasi **identik** dengan training. Solusi: simpan satu `Pipeline`. Catat juga **versi library** — model bisa gagal dimuat di versi berbeda.

---

## 3. Menyajikan Model lewat REST API

Pola paling umum: bungkus model dalam **FastAPI**, terima request JSON, kembalikan prediksi.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="ML Model API")
model = joblib.load("model.joblib")

class InputData(BaseModel):       # validasi skema otomatis
    umur: float
    gaji: float

@app.get("/health")               # health check untuk monitoring/load balancer
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    X = [[data.umur, data.gaji]]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].max()
    return {"prediction": int(pred), "confidence": float(proba)}
```

```bash
uvicorn serve_api:app --reload      # jalankan; buka /docs untuk Swagger UI
```

📂 Versi lengkap & bisa dijalankan: [serve_api.py](./serve_api.py).

**Pola serving lain:** batch (prediksi terjadwal massal), streaming (Kafka), dan untuk model besar/LLM gunakan server khusus (vLLM, TorchServe, Triton).

---

## 4. Kontainerisasi dengan Docker

"Works on my machine" → Docker membuatnya jalan **di mana saja** dengan mengemas kode + dependency + runtime.

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

Dari sini, deploy ke cloud (AWS ECS, Google Cloud Run, Kubernetes) jadi mudah karena unitnya portabel.

---

## 5. Experiment Tracking (MLflow)

Saat kamu menjalankan puluhan eksperimen, kamu **akan** lupa mana yang terbaik & parameternya. Tracking mencatat otomatis: parameter, metrik, dan artefak model.

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 6)

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

# Bandingkan semua run di UI:  mlflow ui  -> http://localhost:5000
```

Manfaat: **reproducibility**, perbandingan eksperimen, dan **model registry** (versioning + tahap staging/production).

---

## 6. Monitoring & Drift

Model bisa "membusuk" diam-diam. Pantau:

| Yang dipantau | Contoh |
|---|---|
| **Operasional** | latency, throughput, error rate, uptime |
| **Data drift** | distribusi input berubah dari data training |
| **Concept drift** | hubungan input→output berubah (perilaku user, ekonomi) |
| **Performa model** | akurasi turun (butuh label aktual) |

```python
# Deteksi data drift sederhana: bandingkan distribusi (uji statistik)
from scipy.stats import ks_2samp
stat, p_value = ks_2samp(fitur_training, fitur_produksi)
if p_value < 0.05:
    print("⚠️ Drift terdeteksi — pertimbangkan re-training")
```

Tools: Evidently AI, Prometheus + Grafana (metrik), WhyLabs. **Prinsip:** log prediksi & input, set alert, dan punya rencana re-training.

---

## 7. CI/CD & Otomatisasi

| Konsep | Untuk ML |
|---|---|
| **CI (Continuous Integration)** | tes kode + validasi data + tes model (mis. akurasi > ambang) |
| **CD (Continuous Deployment)** | deploy otomatis jika lolos tes |
| **CT (Continuous Training)** | re-train terjadwal/terpicu drift |
| **Versioning** | Git untuk kode, **DVC** untuk data, MLflow untuk model |

Strategi deploy aman: **shadow** (model baru jalan paralel tanpa memengaruhi user), **canary** (rilis ke sebagian kecil user), **A/B testing** (bandingkan model lama vs baru pada metrik bisnis).

---

## 8. Skala & Optimasi Inferensi

Saat trafik naik atau model besar:
- **Quantization** — turunkan presisi (FP32→INT8) → lebih kecil & cepat.
- **Distillation** — latih model kecil meniru model besar.
- **ONNX** — format portabel & runtime cepat lintas platform.
- **Batching** — gabungkan request untuk efisiensi GPU.
- **Caching** — simpan hasil untuk input berulang (penting & murah untuk LLM).
- **GPU autoscaling** — skala sesuai beban.

Untuk **LLM** khususnya: gunakan **prompt caching**, streaming response, server khusus (vLLM), dan pantau biaya per token.

---

## Checklist Produksi
- [ ] Pipeline lengkap tersimpan (preprocessing + model), versi library tercatat.
- [ ] API punya `/health`, validasi input, dan penanganan error.
- [ ] Dikemas dalam Docker, lolos build bersih.
- [ ] Eksperimen & versi model terlacak (MLflow).
- [ ] Logging prediksi + input untuk audit & deteksi drift.
- [ ] Monitoring + alert terpasang.
- [ ] Rencana re-training & rollback jelas.

## Latihan
1. Latih model, simpan sebagai `Pipeline`, lalu bangun API dengan [serve_api.py](./serve_api.py). Uji lewat `/docs`.
2. Tulis Dockerfile, build image, dan jalankan API dalam kontainer. Kirim request dengan `curl`.
3. Jalankan 5 eksperimen dengan MLflow tracking. Bandingkan di UI, pilih yang terbaik.
4. Simulasikan drift: ubah distribusi data input dan deteksi dengan uji KS.
5. (Lanjut) Buat workflow GitHub Actions yang menjalankan tes & build Docker otomatis saat push.

➡️ Lanjut: [Modul 08 · Projects & Roadmap Karier](../08-projects-karier/README.md)
