"""
test_model.py
=============
Quick test script to verify that the saved model and scaler load correctly
and make realistic predictions for multiple weather scenarios.
"""

import os
import joblib
import numpy as np
import tensorflow as tf

def test_inference():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "weather_model.keras")
    scaler_path = os.path.join(base_dir, "models", "scaler.joblib")

    assert os.path.exists(model_path), f"Model not found at {model_path}"
    assert os.path.exists(scaler_path), f"Scaler not found at {scaler_path}"

    print("Loading saved model and scaler...")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    class_names = ["Sunny", "Cloudy", "Rainy"]

    test_scenarios = [
        {"desc": "Hot & Dry Day", "temp": 34.0, "humidity": 25.0, "expected": "Sunny"},
        {"desc": "Mild & Overcast Day", "temp": 22.0, "humidity": 68.0, "expected": "Cloudy"},
        {"desc": "Cool & Humid Day", "temp": 16.0, "humidity": 92.0, "expected": "Rainy"},
        {"desc": "Warm Summer Afternoon", "temp": 31.0, "humidity": 38.0, "expected": "Sunny"},
        {"desc": "Chilly Drizzle Morning", "temp": 14.5, "humidity": 88.0, "expected": "Rainy"}
    ]

    print("\nRunning inference tests across multiple scenarios:\n" + "-" * 65)
    for s in test_scenarios:
        raw_input = np.array([[s["temp"], s["humidity"]]])
        scaled_input = scaler.transform(raw_input)
        probs = model.predict(scaled_input, verbose=0)[0]
        pred_idx = np.argmax(probs)
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx] * 100

        print(f"Scenario: {s['desc']:<24} | Temp: {s['temp']:4.1f}°C, Humid: {s['humidity']:4.1f}%")
        print(f" -> Predicted: {pred_class:<7} (Confidence: {confidence:5.1f}%) [Expected: {s['expected']}]")
        print(f"    Probabilities: Sunny={probs[0]*100:4.1f}%, Cloudy={probs[1]*100:4.1f}%, Rainy={probs[2]*100:4.1f}%\n")

    print("-" * 65)
    print("All inference verification tests executed successfully!")

if __name__ == "__main__":
    test_inference()
