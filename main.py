from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List  # <--- NAYI LINE (Batch processing ke liye zaroori)
import joblib
import numpy as np
import pandas as pd 
import os
from fastapi.responses import HTMLResponse # Naya Import
from fastapi.staticfiles import StaticFiles # Naya Import

# 1. Model aur Labels load karo
model = joblib.load("model.pkl")
class_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
app = FastAPI()
from fastapi.staticfiles import StaticFiles # Ye import sabse upar check kar lena

# App define karne ke baad ye line add karo
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Input Guard (Pydantic)
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=15)
    sepal_width: float = Field(..., gt=0, lt=15)
    petal_length: float = Field(..., gt=0, lt=15)
    petal_width: float = Field(..., gt=0, lt=15)

# --- SINGLE PREDICTION (Purana Code) ---
@app.post("/predict")
def predict_flower(data: IrisInput):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    
    prediction_id = int(model.predict(features)[0])
    probability = np.max(model.predict_proba(features)) * 100
    flower_name = class_names.get(prediction_id, "Unknown")

    importances = model.feature_importances_
    importance_dict = {
        "sepal_importance": round(float(importances[0]), 2),
        "petal_importance": round(float(importances[2]), 2)
    }

    log_data = data.model_dump()
    log_data["prediction"] = flower_name
    log_data["confidence"] = f"{round(probability, 2)}%"
    
    df = pd.DataFrame([log_data])
    df.to_csv("prediction_history.csv", mode='a', header=not os.path.exists("prediction_history.csv"), index=False)

    return {
        "flower_name": flower_name,
        "confidence": f"{round(probability, 2)}%",
        "reasoning": importance_dict,
        "message": "Data saved to history.csv"
    }

# --- BATCH PREDICTION (NAYA FUNCTION) --- # <--- NAYI SECTION
@app.post("/predict_batch")
def predict_batch(flowers: List[IrisInput]): # <--- NAYI LINE
    results = [] # Saare answers store karne ke liye khali list
    
    for flower in flowers: # Ek-ek karke saare flowers par loop chalega
        # Features nikaalo
        features = np.array([[flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width]])
        
        # Prediction logic (Same as above)
        prediction_id = int(model.predict(features)[0])
        probability = np.max(model.predict_proba(features)) * 100
        flower_name = class_names.get(prediction_id, "Unknown")
        
        # List mein result add karo
        results.append({
            "flower_name": flower_name,
            "confidence": f"{round(probability, 2)}%"
        })
    
    return {
        "total_processed": len(results),
        "predictions": results
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris AI | Neural Engine</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            :root { --primary: #00ff88; --bg: #0f172a; --card: rgba(30, 41, 59, 0.7); }
            body { 
                font-family: 'Inter', sans-serif; background-color: var(--bg);
                background-image: radial-gradient(circle at 50% 50%, #1e293b 0%, #0f172a 100%);
                color: white; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;
            }
            .glass-card { 
                background: var(--card); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1);
                padding: 2.5rem; border-radius: 24px; width: 450px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            }
            h1 { font-weight: 600; letter-spacing: -1px; margin-bottom: 0.5rem; color: var(--primary); }
            p { color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; }
            .input-field { 
                background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 12px;
                padding: 12px; color: white; font-size: 1rem; transition: 0.2s;
            }
            .input-field:focus { border-color: var(--primary); outline: none; box-shadow: 0 0 0 2px rgba(0,255,136,0.2); }
            button { 
                background: var(--primary); color: #064e3b; border: none; padding: 16px; 
                border-radius: 12px; font-weight: 600; cursor: pointer; width: 100%; transition: 0.3s;
            }
            button:hover { transform: translateY(-2px); opacity: 0.9; box-shadow: 0 10px 20px -5px rgba(0,255,136,0.4); }
            
            #result-section { 
                margin-top: 2rem; display: none; animation: fadeIn 0.6s ease-out;
                border-top: 1px solid #334155; pt: 1.5rem;
            }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            
            .flower-badge { 
                background: rgba(0,255,136,0.1); color: var(--primary); 
                padding: 8px 16px; border-radius: 20px; font-weight: 600; display: inline-block;
            }
            .meter-container { height: 6px; background: #334155; border-radius: 3px; margin: 15px 0; overflow: hidden; }
            .meter-fill { height: 100%; background: var(--primary); width: 0%; transition: 1.5s cubic-bezier(0.1, 0, 0.2, 1); }
            img { width: 100%; border-radius: 16px; margin-top: 15px; filter: grayscale(20%) contrast(110%); }
        </style>
    </head>
    <body>
        <div class="glass-card">
            <h1>Iris Neural Engine</h1>
            <p>Enter biological parameters for instant classification.</p>
            
            <div class="grid">
                <input type="number" id="sl" class="input-field" placeholder="Sepal Length" step="0.1">
                <input type="number" id="sw" class="input-field" placeholder="Sepal Width" step="0.1">
                <input type="number" id="pl" class="input-field" placeholder="Petal Length" step="0.1">
                <input type="number" id="pw" class="input-field" placeholder="Petal Width" step="0.1">
            </div>
            
            <button onclick="predict()">Execute Inference</button>
            
            <div id="result-section">
                <span class="flower-badge" id="name"></span>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 10px;">Classification Confidence</div>
                <div class="meter-container"><div id="fill" class="meter-fill"></div></div>
                <div id="conf" style="font-weight: 600;"></div>
                <img id="pic" src="" alt="prediction">
            </div>
        </div>

        <script>
    async function predict() {
        const data = {
            sepal_length: document.getElementById('sl').value,
            sepal_width: document.getElementById('sw').value,
            petal_length: document.getElementById('pl').value,
            petal_width: document.getElementById('pw').value
        };
        
        const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        
        const out = await res.json();
        
        // UI Update logic
        document.getElementById('result-section').style.display = 'block';
        document.getElementById('name').innerText = "SPECIES: " + out.flower_name.toUpperCase();
        document.getElementById('conf').innerText = out.confidence;
        document.getElementById('fill').style.width = out.confidence;

        // FIXED PHOTO LINKS (Ab Rose nahi aayega!)
        const img = document.getElementById('pic');
        const links = {
                "Setosa": "static/images/setosa.jpg",
                "Versicolor": "static/images/versicolor.jpg",
                "Virginica": "static/images/virginica.jpg"
        };
        
        img.src = links[out.flower_name] || ""; 
    }
    </script> 
    </body>
    </html>
    """