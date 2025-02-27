from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("model/ckd_model.pkl")

# Define API route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()

        # Convert data into numpy array for model
        input_features = np.array(data["features"]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1] * 100

        # Return response
        return jsonify({
            "prediction": int(prediction),
            "probability": round(probability, 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
