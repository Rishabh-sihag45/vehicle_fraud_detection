import joblib
import numpy as np

# Load trained model
model = joblib.load("fraud_model.pkl")

# Example claim data
data = np.array([[10,35,123456,2015,1,2,500,1200,1,2,3,1,1,2,15000,1]])

prediction = model.predict(data)

if prediction == 1:
    print("Fraud Claim Detected")
else:
    print("Genuine Claim")