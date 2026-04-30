import logging
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List 
import joblib
import numpy as np
import os
import time
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles 

# --- 1. SETUP & AUTH ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_KEY = "admin"
api_key_header = APIKeyHeader(name="X-API-KEY")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")
    return api_key

start_time = time.time()
model = joblib.load("models/v1/model.pkl") 
class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}

app = FastAPI(title="Iris Neural Engine", version="2.6.0")

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=15)
    sepal_width: float = Field(..., gt=0, lt=15)
    petal_length: float = Field(..., gt=0, lt=15)
    petal_width: float = Field(..., gt=0, lt=15)

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iris AI | Neural Engine</title>
    <style>
        :root { --primary: #00ff88; --bg: #0f172a; --card: rgba(30, 41, 59, 0.9); }
        body { 
            font-family: 'Segoe UI', sans-serif; background-color: var(--bg);
            background-image: radial-gradient(circle at 50% 50%, #1e293b 0%, #0f172a 100%);
            color: white; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0;
        }
        .glass-card { 
            background: var(--card); backdrop-filter: blur(15px); border: 1px solid rgba(255,255,255,0.1);
            padding: 2.5rem; border-radius: 28px; width: 500px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.6);
        }
        .header-section { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .key-input { 
            background: rgba(15, 23, 42, 0.8); border: 1px solid #334155; border-radius: 6px;
            padding: 5px; color: var(--primary); font-size: 0.7rem; width: 80px; text-align: center;
        }
        h1 { color: var(--primary); margin: 0; font-size: 1.5rem; }
        .toggle-btn { 
            background: rgba(51, 65, 85, 0.4); color: var(--primary); border: 1px solid rgba(0,255,136,0.3);
            padding: 8px 16px; border-radius: 8px; cursor: pointer; display: block; margin: 0 auto 20px; font-size: 0.7rem;
        }
        .input-field { 
            background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 12px;
            padding: 12px; color: white; margin-bottom: 10px; width: 92%; font-size: 0.9rem;
        }
        button.exec-btn { 
            background: var(--primary); color: #064e3b; border: none; padding: 16px; 
            border-radius: 14px; font-weight: 700; cursor: pointer; width: 100%; margin-top: 10px;
        }
 
        .res-img { 
            width: 100%; 
            height: 100px; /* Force consistent height */
            object-fit: cover; /* This crops the image to fit the box without stretching */
            border-radius: 10px; 
            margin-top: 8px; 
       }

        /* BATCH CARD BOXES - RESTORED */
        #result-section { margin-top: 1.5rem; border-top: 1px solid #334155; padding-top: 1rem; display: none; }
        .batch-grid { 
            display: grid; grid-template-columns: 1fr 1fr; gap: 12px; 
            max-height: 300px; overflow-y: auto; padding-right: 5px;
        }
        .batch-card { 
            background: rgba(255, 255, 255, 0.05); 
            border: 1px solid rgba(0, 255, 136, 0.2); 
            border-radius: 12px; 
            padding: 10px; 
            text-align: center;
            transition: 0.3s;
        }
        .batch-card:hover { border-color: var(--primary); background: rgba(0, 255, 136, 0.05); }
        .res-img { width: 100%; border-radius: 8px; margin-top: 8px; height: 80px; object-fit: cover; }
    </style>
</head>
<body>
    <div class="glass-card">
        <div class="header-section">
            <h1>Neural Engine</h1>
            <input type="password" id="api-key" class="key-input" placeholder="Auth Key">
        </div>

        <button class="toggle-btn" onclick="toggleMode()">MODE: SINGLE INFERENCE</button>

        <div id="single-inputs">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <input type="number" id="sl" class="input-field" placeholder="Sepal L" value="5.1">
                <input type="number" id="sw" class="input-field" placeholder="Sepal W" value="3.5">
                <input type="number" id="pl" class="input-field" placeholder="Petal L" value="1.4">
                <input type="number" id="pw" class="input-field" placeholder="Petal W" value="0.2">
            </div>
        </div>

        <div id="batch-inputs" style="display:none;">
            <textarea id="batch-json" class="input-field" rows="5" placeholder='Paste JSON List here...'></textarea>
        </div>

        <button class="exec-btn" onclick="predict()">EXECUTE ANALYSIS</button>
        <div id="result-section"></div>
    </div>

    <script>
    let isBatch = false;
    function toggleMode() {
        isBatch = !isBatch;
        document.getElementById('single-inputs').style.display = isBatch ? 'none' : 'block';
        document.getElementById('batch-inputs').style.display = isBatch ? 'block' : 'none';
        document.querySelector('.toggle-btn').innerText = isBatch ? 'MODE: BATCH ANALYSIS' : 'MODE: SINGLE INFERENCE';
    }

    async function predict() {
        const key = document.getElementById('api-key').value;
        const endpoint = isBatch ? '/predict/batch' : '/predict';
        let body;

        if(isBatch) {
            try { body = JSON.parse(document.getElementById('batch-json').value); }
            catch(e) { alert("Invalid JSON format!"); return; }
        } else {
            const vals = [
                document.getElementById('sl').value, 
                document.getElementById('sw').value, 
                document.getElementById('pl').value, 
                document.getElementById('pw').value
            ];
            if(vals.some(v => v === "")) { alert("Please fill all fields"); return; }
            body = { 
                sepal_length: parseFloat(vals[0]), 
                sepal_width: parseFloat(vals[1]), 
                petal_length: parseFloat(vals[2]), 
                petal_width: parseFloat(vals[3]) 
            };
        }

        const res = await fetch(endpoint, {
            method: 'POST',
            headers: {'Content-Type': 'application/json', 'X-API-KEY': key},
            body: JSON.stringify(body)
        });

        if (res.status === 403) { alert("Access Denied: Invalid Key"); return; }
        
        const data = await res.json();
        const resDiv = document.getElementById('result-section');
        resDiv.style.display = 'block';
        resDiv.innerHTML = '';

        if(!isBatch) {
            // Updated single result template for a LARGE image
            resDiv.innerHTML = `
                <div style="text-align:center; padding: 10px;">
                    <div style="background:rgba(0,255,136,0.1); color:var(--primary); padding:8px 20px; border-radius:20px; font-weight:bold; display:inline-block; margin-bottom:10px;">
                        ${data.flower_name.toUpperCase()}
                    </div>
                    <p style="font-size:1.1rem; color:#94a3b8; margin:5px 0;">Confidence: <strong>${data.confidence}</strong></p>
                    <img src="${data.image_url}" style="width:100%; max-width:350px; border-radius:18px; margin-top:15px; border: 2px solid var(--primary); box-shadow: 0 0 20px rgba(0,255,136,0.2);">
                </div>`;
        } else {
            let html = '<div class="batch-grid">';
            data.predictions.forEach(p => {
                html += `
                    <div class="batch-card">
                        <div style="color:var(--primary); font-weight:bold; font-size:0.75rem;">${p.flower_name.toUpperCase()}</div>
                        <div style="font-size:0.65rem; color:#94a3b8;">${p.confidence}</div>
                        <img src="${p.image_url}" class="res-img">
                    </div>`;
            });
            resDiv.innerHTML = html + '</div>';
        }
    }
    </script>
</body>
</html>
"""

# --- 3. ENDPOINTS ---

@app.post("/predict")
def predict_single(data: IrisInput, key: str = Depends(verify_api_key)):
    logger.info(f"Single Prediction Request")
    feat = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    pid = int(model.predict(feat)[0])
    name = class_names[pid]
    prob = np.max(model.predict_proba(feat)) * 100
    return {"flower_name": name, "confidence": f"{round(prob, 2)}%", "image_url": f"/static/images/{name}.jpg"}

@app.post("/predict/batch")
def predict_batch(flowers: List[IrisInput], key: str = Depends(verify_api_key)):
    logger.info(f"Batch Request: {len(flowers)} items")
    results = []
    for f in flowers:
        feat = np.array([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]])
        pid = int(model.predict(feat)[0])
        name = class_names[pid]
        prob = np.max(model.predict_proba(feat)) * 100
        results.append({"flower_name": name, "confidence": f"{round(prob, 2)}%", "image_url": f"/static/images/{name}.jpg"})
    return {"predictions": results}

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_CONTENT