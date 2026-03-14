import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from house_price_prediction.preprocessing import create_inference_df

app = FastAPI(title="Aura Estates Predictor")

# Mount static files
app.mount("/static", StaticFiles(directory="src/house_price_prediction/webapp/static"), name="static")

# Templates
templates = Jinja2Templates(directory="src/house_price_prediction/webapp/templates")

# Model Loading
MODEL_PATH = "models/xgboost_model.joblib"
FEATURES_PATH = "models/feature_names.joblib"

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
        return {"error": "Model not trained yet. Run the training script first."}
    
    # Load model and feature list
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    
    # Create input vector
    input_data = features.model_dump()
    input_df = create_inference_df(input_data, feature_names)
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    return {"prediction": float(prediction)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
