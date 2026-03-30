from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Load the "brain" (the model you just trained)
model = joblib.load("model.pkl")

# 2. Create the App
app = FastAPI()

# 3. Define what the input should look like (4 numbers for Iris)
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 4. Create the "Predict" endpoint
@app.post("/predict")
def predict_flower(data: IrisInput):
    # Convert input to a format the model understands
    features = np.array([[
        data.sepal_length, 
        data.sepal_width, 
        data.petal_length, 
        data.petal_width
    ]])
    
    # Get the prediction (0, 1, or 2)
    prediction = model.predict(features)
    
    # Return as JSON
    return {"prediction": int(prediction[0])}