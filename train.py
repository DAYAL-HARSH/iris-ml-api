import joblib
from xgboost import XGBClassifier
from sklearn.datasets import load_iris

# Load the data
data = load_iris()
X, y = data.data, data.target

# Create and train the model
model = XGBClassifier()
model.fit(X, y)

# Save the model as a file named "model.pkl"
joblib.dump(model, "model.pkl")
print("Success! model.pkl has been created.")