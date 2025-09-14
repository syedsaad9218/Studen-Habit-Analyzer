from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load saved model, scaler, and label encoder
model_bundle = joblib.load("models/model_v1.joblib")
model = model_bundle["model"]
scaler = model_bundle["scaler"]
label_encoder = model_bundle["label_encoder"]

@app.route("/")
def home():
    return render_template("index.html")  # Simple form for input

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        study_hours = float(request.form["study_hours"])
        sleep_hours = float(request.form["sleep_hours"])
        phone_usage = float(request.form["phone_usage"])
        exercise_time = float(request.form["exercise_time"])
        attendance = float(request.form["attendance"])

        # Put into numpy array (1 row, 5 columns)
        input_data = np.array([[study_hours, sleep_hours, phone_usage, exercise_time, attendance]])

        # Scale input (important for Logistic Regression, RF ignores but we keep consistent)
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)[0]  # RF used here
        prediction_label = label_encoder.inverse_transform([prediction])[0]

        return render_template("index.html", result=f"Predicted Performance: {prediction_label}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
