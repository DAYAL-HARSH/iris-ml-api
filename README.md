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

2. **Virtual Environment & Libraries**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. **Server Start karein**
   ```bash
   uvicorn main:app --reload

# 📊 API Documentation

1. **Single Prediction**
   Endpoint: POST /predict
   Sample Request (JSON):
   ```JSON
   {
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
  } 