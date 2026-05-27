"""
Contoh menyajikan model ML lewat REST API dengan FastAPI.

Demo ini melatih model kecil saat startup (agar self-contained). Di produksi,
kamu MEMUAT pipeline tersimpan (joblib.load), bukan melatih ulang.

Jalankan:
    pip install fastapi uvicorn scikit-learn joblib
    uvicorn serve_api:app --reload
    # Buka http://localhost:8000/docs untuk UI interaktif (Swagger)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL = {}   # tempat menyimpan model setelah dimuat/dilatih


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: di produksi -> MODEL["clf"] = joblib.load("model.joblib") ---
    data = load_breast_cancer()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
    pipe.fit(data.data, data.target)
    MODEL["clf"] = pipe
    MODEL["n_features"] = data.data.shape[1]
    MODEL["target_names"] = list(data.target_names)
    print(f"Model siap. Mengharapkan {MODEL['n_features']} fitur.")
    yield
    MODEL.clear()   # cleanup saat shutdown


app = FastAPI(title="ML Model API", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    # Validasi otomatis: harus list float dengan panjang benar
    features: list[float] = Field(..., description="Vektor fitur input")


class PredictResponse(BaseModel):
    prediction: int
    label: str
    confidence: float


@app.get("/health")
def health():
    """Health check untuk load balancer / monitoring."""
    return {"status": "ok", "model_loaded": "clf" in MODEL}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if "clf" not in MODEL:
        raise HTTPException(status_code=503, detail="Model belum siap")

    if len(req.features) != MODEL["n_features"]:
        raise HTTPException(
            status_code=422,
            detail=f"Butuh {MODEL['n_features']} fitur, dapat {len(req.features)}",
        )

    X = [req.features]
    pred = int(MODEL["clf"].predict(X)[0])
    proba = float(MODEL["clf"].predict_proba(X)[0].max())
    return PredictResponse(
        prediction=pred,
        label=MODEL["target_names"][pred],
        confidence=round(proba, 4),
    )
