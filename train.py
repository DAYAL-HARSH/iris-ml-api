import joblib
from xgboost import XGBClassifier
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

model = XGBClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Success! model.pkl has been created.")