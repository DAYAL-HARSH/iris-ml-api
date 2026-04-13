# 🚀 Live Demo: https://iris-predictor-harsh.onrender.com/

# 🌸 Iris Neural Engine: Production-Grade ML API

Iris Species Predictor: A high-performance Machine Learning web application leveraging XGBoost for precision classification. Features a sleek, premium Glassmorphism UI and a lightning-fast FastAPI backend integration.

## 🚀 Project Overview
This project goes beyond simple inference, demonstrating a production-grade End-to-End ML Pipeline. It integrates Pydantic for rigorous data validation, automated logging for prediction traceability, and integrated model explainability to provide insights into feature importance.

## 🛠️ Tech Stack
* **Backend:** FastAPI (Python)
* **Machine Learning:** XGBoost, Scikit-learn
* **Frontend:** HTML5, CSS3 (Glassmorphism), JavaScript (Fetch API)
* **Data Handling:** Pandas, NumPy
* **Model Management:** Joblib

## ⚙️ How to Run

1. **How to Clone Respository:**
   ```bash
   git clone https://github.com/DAYAL-HARSH/iris-ml-api.git
   cd iris-ml-api
   ```

2. **Virtual Environment & Libraries**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **How to start Server**
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

## 📂 Folder Structure

```text
iris-ml-api/
├── static/
│   └── images/            # Assets for species visualization and UI enhancement
├── main.py                # Core FastAPI application containing API logic and UI routes
├── model.pkl              # Serialized XGBoost classification model
├── prediction_history.csv # Persistent log for tracking and predictions
└── requirements.txt       # Project dependencies and environment specifications
```


## 🐳 Docker Deployment
To run this application using Docker:

1. **Build the image:**
   ```bash
   docker build -t iris-ml-api .
   ```
2. **Run the Container:**
   ```bash
   docker run -p 8000:8000 iris-ml-api
   ```
   
