from sklearn.ensemble import IsolationForest
from data import generate_data
import joblib

data = generate_data()

model = IsolationForest(contamination=0.1)
model.fit(data)

# Save model
joblib.dump(model, "fraud_model.pkl")

print("Model trained and saved")