import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client

from preprocessing import FEATURE_ORDER, preprocess_and_measure
from model_router import load_bundle, available_models

APP_VERSION = "ecoexposure-inference-v1"

SUPABASE_URL = os.getenv("SUPABASE_URL","").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY","").strip()
ALLOWED_ORIGINS = [
    x.strip() for x in os.getenv(
        "ALLOWED_ORIGINS",
        "https://melinda-ecoteraasia.github.io,http://localhost:8000,http://127.0.0.1:8000"
    ).split(",") if x.strip()
]

app = FastAPI(title="EcoExposure Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    sample_id: str
    water_type: str
    model_name: str
    top_photo_path: str
    storage_bucket: str = "ecoexposure_images"

def get_supabase():
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(
            status_code=503,
            detail="Supabase backend credentials are not configured on Render."
        )
    return create_client(SUPABASE_URL,SUPABASE_SERVICE_ROLE_KEY)

@app.get("/")
def root():
    return {
        "service":"EcoExposure Inference API",
        "version":APP_VERSION,
        "status":"ok",
        "models":available_models(),
    }

@app.get("/health")
def health():
    return {
        "status":"ok",
        "version":APP_VERSION,
        "models":available_models(),
        "supabase_configured":bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
    }

@app.post("/predict")
def predict(request:PredictRequest,authorization:Optional[str]=Header(default=None)):
    try:
        bundle=load_bundle(request.model_name)
    except KeyError as exc:
        raise HTTPException(status_code=400,detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503,detail=str(exc))

    sb=get_supabase()
    try:
        image_bytes=sb.storage.from_(request.storage_bucket).download(request.top_photo_path)
    except Exception as exc:
        raise HTTPException(status_code=502,detail=f"Storage download failed: {exc}")

    try:
        metrics,qc=preprocess_and_measure(image_bytes,filename=request.top_photo_path)
    except Exception as exc:
        raise HTTPException(status_code=422,detail=f"Image preprocessing failed: {exc}")

    feature_names=bundle.get("feature_names") or FEATURE_ORDER
    model=bundle["model"]
    X=pd.DataFrame([{name:metrics.get(name) for name in feature_names}],columns=feature_names)

    try:
        prediction=model.predict(X)[0]
    except Exception as exc:
        raise HTTPException(status_code=500,detail=f"Prediction failed: {exc}")

    confidence=None
    class_probabilities=None
    if hasattr(model,"predict_proba"):
        try:
            probs=model.predict_proba(X)[0]
            classes=[str(x) for x in model.classes_]
            confidence=float(np.max(probs))
            class_probabilities={classes[i]:float(probs[i]) for i in range(len(probs))}
        except Exception:
            pass

    try:
        predicted_numeric=int(float(prediction))
    except Exception:
        predicted_numeric=None

    index=None; band=None
    if predicted_numeric is not None:
        if predicted_numeric<=25:
            index,band=1,"Green"
        elif predicted_numeric<=50:
            index,band=2,"Green-Yellow"
        elif predicted_numeric<=75:
            index,band=3,"Yellow"
        elif predicted_numeric<=100:
            index,band=4,"Orange"
        else:
            index,band=5,"Red"

    return {
        "sample_id":request.sample_id,
        "water_type":request.water_type,
        "model_name":request.model_name,
        "model_version":bundle.get("model_version",request.model_name),
        "predicted_mp":predicted_numeric if predicted_numeric is not None else str(prediction),
        "ecoexposure_index":index,
        "band":band,
        "confidence":confidence,
        "class_probabilities":class_probabilities,
        "metrics":metrics,
        "qc":qc,
        "api_version":APP_VERSION,
    }
