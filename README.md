# 🚀 ML Inference API with FastAPI (Iris Classification)

## 📌 Overview

This project is a **containerized machine learning inference service** built using FastAPI and Docker. It exposes REST APIs for real-time and batch predictions using a trained XGBoost model.

The goal of this project is to simulate a **production-ready ML backend system**, focusing on API design, deployment, and scalability rather than just model training.

---

## ⚙️ Tech Stack

* **Backend:** FastAPI (Python)
* **Machine Learning:** XGBoost, Scikit-learn
* **Containerization:** Docker
* **Deployment:** Render
* **Language:** Python

---

## 🧠 Problem Statement

The system predicts the species of an iris flower based on input features:

* Sepal length
* Sepal width
* Petal length
* Petal width

While the dataset is simple, this project emphasizes **building a robust ML service**, not just model accuracy.

---

## 🏗️ System Architecture

Client → FastAPI → ML Model → Response

---

## 🔌 API Endpoints

### 1. Predict (Single Input)

`POST /predict`

**Input:**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Output:**

```json
{
  "prediction": "setosa"
}
```

---

### 2. Batch Prediction

`POST /batch_predict`

* Accepts multiple inputs
* Returns predictions for all records

---

### 3. Health Check

`GET /health`

* Checks if the API is running

---

## 🧪 Model Details

* Algorithm: XGBoost Classifier
* Dataset: Iris Dataset
* Evaluation Metrics: Accuracy, Precision, Recall

---

## 🐳 Docker Support

The application is fully containerized:

```bash
docker build -t iris-api .
docker run -p 8000:8000 iris-api
```

---

## 🌐 Deployment

The API is deployed and accessible online:

👉 https://iris-predictor-harsh.onrender.com

---

## 📈 Future Improvements

* Model versioning (`/v1`, `/v2` endpoints)
* Logging and monitoring
* Input validation enhancements
* CI/CD pipeline integration

---

## 💡 Key Learnings

* Designing REST APIs for ML inference
* Containerizing applications using Docker
* Deploying backend services to cloud platforms
* Structuring scalable ML systems

---
