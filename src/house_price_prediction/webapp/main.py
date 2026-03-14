import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from house_price_prediction.preprocessing import transform_inference_data

app = FastAPI(title="Aura Estates Predictor")

# Mount static files
app.mount("/static", StaticFiles(directory="src/house_price_prediction/webapp/static"), name="static")

# Templates
templates = Jinja2Templates(directory="src/house_price_prediction/webapp/templates")

# Model Loading
# Model Loading
MODEL_PATH = "models/xgboost_model.joblib"
TRANSFORMERS_PATH = "models/transformers.joblib"

class HouseFeatures(BaseModel):
    GrLivArea: float
    OverallQual: int
    TotalBsmtSF: float
    FullBath: int
    GarageCars: int
    YearBuilt: int

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(features: HouseFeatures):
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained yet."}
    
    # Load model and transformers
    model = joblib.load(MODEL_PATH)
    transformers = joblib.load(TRANSFORMERS_PATH)
    
    # Create input vector
    input_data = features.model_dump()
    input_df = transform_inference_data(input_data, transformers)
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    return {"prediction": float(prediction)}

def start():
    """Launched with `uv run serve-app`"""
    import uvicorn
    uvicorn.run("house_price_prediction.webapp.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()
