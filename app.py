from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()

# ✅ Fix model path for deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")

model = joblib.load(MODEL_PATH)

# ✅ Input schema
class InputData(BaseModel):
    data: list

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(input: InputData):
    if len(input.data) != 6:
        return {"error": "Invalid input. Expected 6 features"}

    arr = np.array(input.data).reshape(1, -1)

    prediction = model.predict(arr)[0]
    score = model.decision_function(arr)[0]

    return {
        "prediction": "fraud" if prediction == -1 else "normal",
        "score": float(score)
    }

# Optional (not required for Render but fine to keep)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)