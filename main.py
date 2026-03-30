from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd  # Naya: Excel-jaisi file banane ke liye
import os            # Naya: File check karne ke liye

# 1. Model aur Labels load karo
model = joblib.load("model.pkl")
class_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
app = FastAPI()

# 2. Input Guard (Pydantic)
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=15)
    sepal_width: float = Field(..., gt=0, lt=15)
    petal_length: float = Field(..., gt=0, lt=15)
    petal_width: float = Field(..., gt=0, lt=15)

@app.post("/predict")
def predict_flower(data: IrisInput):
    # Data ko matrix mein badlo
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    
    # Prediction aur Confidence nikaalo
    prediction_id = int(model.predict(features)[0])
    probability = np.max(model.predict_proba(features)) * 100
    flower_name = class_names.get(prediction_id, "Unknown")

    # --- NAYA FEATURE 1: FEATURE IMPORTANCE (Detective) ---
    # Model se pucho: "Tune kis feature ko kitna importance diya?"
    importances = model.feature_importances_
    importance_dict = {
        "sepal_importance": round(float(importances[0]), 2),
        "petal_importance": round(float(importances[2]), 2)
    }

    # --- NAYA FEATURE 2: HISTORY LOGGING (Register) ---
    # Ek chhota sa record banao
    log_data = data.model_dump() # User ka input
    log_data["prediction"] = flower_name
    log_data["confidence"] = f"{round(probability, 2)}%"
    
    # Isse "prediction_history.csv" mein save karo
    df = pd.DataFrame([log_data])
    # 'a' mode ka matlab hai purane data ke niche naya line add karo (Append)
    df.to_csv("prediction_history.csv", mode='a', header=not os.path.exists("prediction_history.csv"), index=False)

    return {
        "flower_name": flower_name,
        "confidence": f"{round(probability, 2)}%",
        "reasoning": importance_dict,
        "message": "Data saved to history.csv"
    }