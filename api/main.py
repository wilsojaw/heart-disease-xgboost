from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import sys
import os

# Add src/ to the Python path
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import preprocess_data

app = FastAPI()

# Load model once at startup
model = joblib.load("../model/model.pkl")

# Define input schema
class PatientData(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])
        print("Received input:", input_df)

        # Let the model pipeline handle preprocessing and prediction
        prediction = model.predict(input_df)[0]
        return {"prediction": int(prediction)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}