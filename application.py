import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from config.paths_config import MODEL_OUTPUT_PATH

app = Flask(__name__)
loaded_model = joblib.load(MODEL_OUTPUT_PATH)  # Ensure model.pkl exists

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        lead_time = int(data["lead_time"])
        no_of_special_request = int(data["no_of_special_request"])
        avg_price_per_room = float(data["avg_price_per_room"])
        arrival_month = int(data["arrival_month"])
        arrival_date = int(data["arrival_date"])
        market_segment_type = int(data["market_segment_type"])
        no_of_week_nights = int(data["no_of_week_nights"])
        no_of_weekend_nights = int(data["no_of_weekend_nights"])
        type_of_meal_plan = int(data["type_of_meal_plan"])
        room_type_reserved = int(data["room_type_reserved"])

        features = np.array([[
            lead_time, no_of_special_request, avg_price_per_room,
            arrival_month, arrival_date, market_segment_type,
            no_of_week_nights, no_of_weekend_nights, type_of_meal_plan,
            room_type_reserved
        ]])
        
        prediction = loaded_model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    
    except ValueError:
        return jsonify({"error": "Invalid input. Please enter valid values."}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
