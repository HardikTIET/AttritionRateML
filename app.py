import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Define categorical columns and load LabelEncoders
categorical_cols = ['BusinessTravel', 'Department', 'MaritalStatus']
label_encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in categorical_cols}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Retrieve form inputs
        input_data = {
            "BusinessTravel": request.form["BusinessTravel"],
            "Department": request.form["Department"],
            "MaritalStatus": request.form["MaritalStatus"],
            "TotalWorkingYears": int(request.form["TotalWorkingYears"]),
            "TrainingTimesLastYear": int(request.form["TrainingTimesLastYear"]),
            "YearsWithCurrManager": int(request.form["YearsWithCurrManager"]),
            "EnvironmentSatisfaction": int(request.form["EnvironmentSatisfaction"]),
            "JobSatisfaction": int(request.form["JobSatisfaction"]),
            "WorkLifeBalance": int(request.form["WorkLifeBalance"]),
            "JobInvolvement": int(request.form["JobInvolvement"])
        }

        # Encode categorical values
        for col in categorical_cols:
            if input_data[col] not in label_encoders[col].classes_:
                # Print available classes to debug
                print(f"Available classes for {col}: {label_encoders[col].classes_}")

                # Handle unknown category dynamically
                label_encoders[col].classes_ = np.append(label_encoders[col].classes_, input_data[col])
            
            input_data[col] = label_encoders[col].transform([input_data[col]])[0]

        # Convert input into a NumPy array and reshape
        features = np.array(list(input_data.values())).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        # Convert prediction to meaningful label
        result = "Employee is likely to leave ❌" if prediction == 1 else "Employee is likely to stay ✅"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
