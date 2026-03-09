from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = [float(x) for x in request.form.values()]
    data = np.array([data])

    prediction = model.predict(data)

    if prediction == 1:
        result = "Fraud Claim Detected"
    else:
        result = "Genuine Claim"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5001, debug=True)