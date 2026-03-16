from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Aether DS Inference API")

class InferenceRequest(BaseModel):
    features: list[float]

class InferenceResponse(BaseModel):
    prediction: int
    probability: list[float]

# Load model at startup
MODEL_PATH = 'models/model.joblib'
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        # Fallback or error
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        data = np.array(request.features).reshape(1, -1)
        prediction = int(model.predict(data)[0])
        probability = model.predict_proba(data)[0].tolist()
        
        return InferenceResponse(prediction=prediction, probability=probability)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
