🌸 Iris Neural Engine: Production-Grade ML API

Ek high-performance Machine Learning Web Application jo XGBoost ka use karke Iris flower ki species predict karti hai. Isme ek premium Glassmorphism UI aur FastAPI backend ka integration hai.

🚀 Project Overview

Ye project sirf ek basic model prediction nahi hai, balki ek complete End-to-End ML Pipeline ka demonstration hai. Isme input validation (Pydantic), feature importance reasoning, aur automated prediction history logging jaise advanced features include kiye gaye hain.

🛠️ Tech Stack
Backend: FastAPI (Python)

Machine Learning: XGBoost, Scikit-learn

Frontend: HTML5, CSS3 (Glassmorphism), JavaScript (Fetch API)

Data Handling: Pandas, NumPy

Model Management: Joblib

⚙️ How to Run
Repository Clone karein:

Bash
git clone https://github.com/your-username/iris-ml-api.git
cd iris-ml-api

Virtual Environment banayein aur Libraries install karein:

Bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

Server Start karein:

Bash
uvicorn main:app --reload
Access karein:

Web UI: http://127.0.0.1:8000/

API Docs (Swagger): http://127.0.0.1:8000/docs

📊 API Documentation

1. Single Prediction
Endpoint: POST /predict

Sample Request (JSON):

JSON
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
Sample Response (JSON):

JSON
{
  "flower_name": "Setosa",
  "confidence": "99.85%",
  "reasoning": {
    "sepal_importance": 0.15,
    "petal_importance": 0.85
  },
  "message": "Data saved to history.csv"
}


2. Batch Prediction
Endpoint: POST /predict_batch

Sample Request (JSON List):

JSON
[
  { "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2 },
  { "sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4 }
]

📂 Folder Structure
Plaintext
iris-ml-api/
├── main.py              # FastAPI Backend & Integrated UI
├── model.pkl            # Trained XGBoost Model Artifact
├── prediction_history.csv # Automated Logging Ledger
├── requirements.txt     # Dependency List
└── static/
    └── images/          # Local Flower Assets (Optimized)