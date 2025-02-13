import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained churn model
with open("churn_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract 19 feature values from the form
        features = [float(request.form[f'feature{i+1}']) for i in range(19)]
        input_array = np.array([features])

        # Make prediction
        prediction = model.predict(input_array)[0]
        result = "Customer is likely to churn" if prediction == 1 else "Customer is not likely to churn"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
