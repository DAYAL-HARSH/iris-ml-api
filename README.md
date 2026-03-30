# 🌸 Iris Flower Classification API

Ek simple aur powerful ML API jo flower ki measurements lekar batati hai ki wo kaunsa species hai. 

## 🚀 Tech Stack
* **Language:** Python
* **Model:** XGBoost (Gradient Boosting)
* **API Framework:** FastAPI
* **Validation:** Pydantic
* **Deployment Tool:** Uvicorn

## 🛠️ Project Structure
* `train.py`: Model training aur artifact (`model.pkl`) banane ke liye.
* `main.py`: FastAPI server jo predictions handle karta hai.
* `model.pkl`: Trained model ka binary format (Artifact).

## 🏃 How to Run
1. Environment activate karein:
   ```bash
   conda activate base
2. server start karein
   ```bash 
   uvicorn main:app --reload
3. Browser mein open karein: http://127.0.0.1:8000/docs