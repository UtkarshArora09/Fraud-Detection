from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("fraud_model.pkl")

# ✅ Define input schema
class InputData(BaseModel):
    data: list

@app.post("/predict")
def predict(input: InputData):
    arr = np.array(input.data).reshape(1, -1)

    prediction = model.predict(arr)[0]
    score = model.decision_function(arr)[0]

    return {
        "prediction": "fraud" if prediction == -1 else "normal",
        "score": float(score)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)