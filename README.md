# 🌸 Iris Neural Engine: Production-Grade ML API

Ek high-performance Machine Learning Web Application jo **XGBoost** ka use karke Iris flower ki species predict karti hai. Isme ek premium **Glassmorphism UI** aur **FastAPI** backend ka integration hai.

## 🚀 Project Overview
Ye project sirf ek basic model prediction nahi hai, balki ek complete **End-to-End ML Pipeline** ka demonstration hai. Isme advanced features include kiye gaye hain:
* **Input Validation:** Pydantic models ka use karke.
* **Explainability:** Feature importance reasoning.
* **Logging:** Automated prediction history ledger.

## 🛠️ Tech Stack
* **Backend:** FastAPI (Python)
* **Machine Learning:** XGBoost, Scikit-learn
* **Frontend:** HTML5, CSS3 (Glassmorphism), JavaScript (Fetch API)
* **Data Handling:** Pandas, NumPy
* **Model Management:** Joblib

## ⚙️ How to Run

1. **Repository Clone karein:**
   ```bash
   git clone [https://github.com/your-username/iris-ml-api.git](https://github.com/your-username/iris-ml-api.git)
   cd iris-ml-api
   ```

2. **Virtual Environment & Libraries**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Server Start karein**
   ```bash
   uvicorn main:app --reload
   ```


## 📊 API Documentation

### 1. Single Prediction (`POST /predict`)

**Sample Request:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Sample Response:**
```json
{
  "flower_name": "Setosa",
  "confidence": "99.85%",
  "message": "Data saved to history.csv"
}
```

### 2. Batch Prediction (POST /predict_batch)

**Sample Request:**
```json
[
  { "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2 },
  { "sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4 }
]
```

## 📂 Folder Structure

```text
iris-ml-api/
├── main.py                # FastAPI Backend aur Integrated UI Logic
├── model.pkl              # Trained XGBoost Model file
├── prediction_history.csv # CSV file jahan saara prediction data log hota hai
├── requirements.txt       # Project ki saari Python dependencies
└── static/                # Static files ka folder
    └── images/            # Local flower images jo UI mein dikhti hain