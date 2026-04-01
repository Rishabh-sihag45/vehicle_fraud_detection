from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("fraud_model.pkl")
encoders = joblib.load("encoders.pkl")   # 🔥 important

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get raw input (strings + numbers)
    form_data = request.form.to_dict()

    # Convert to DataFrame
    df = pd.DataFrame([form_data])

    # Convert numeric columns
    numeric_cols = [
        'f1','f2','f3','f4','f7','f8','f15'
    ]

    for col in numeric_cols:
        df[col] = df[col].astype(float)

    # 🔥 Encode string columns
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))

    # Prediction
    prediction = model.predict(df)[0]

    result = "Fraud Claim Detected ❌" if prediction == 1 else "Genuine Claim ✅"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
