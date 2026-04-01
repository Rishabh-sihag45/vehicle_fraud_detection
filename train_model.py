import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("insurance_claims.csv")

# 🔥 Store encoders for each column
encoders = {}

# Convert text to numbers
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        encoders[column] = le   # ✅ store encoder

# Features and target
X = data.drop("fraud_reported", axis=1)
y = data["fraud_reported"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# ✅ Save both
joblib.dump(model, "fraud_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Model + Encoders saved successfully 🚀")
