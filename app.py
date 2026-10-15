
import os
import io
import requests
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from googletrans import Translator
from PIL import Image

app = Flask(__name__)
CORS(app)

# ---------- LOAD MODEL AND DATA ----------
model = load_model('plant_disease_model.h5')
fertilizer_df = pd.read_csv('fertilizer_data.csv')

class_labels = [
    'Aphids_Disease',
    'Blotch',
    'Healthy_Leaf',
    'Leaf_Spot'
]

translator = Translator()

# ---------- TELEGRAM CONFIG ----------
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID", "5982626821")
TARGET_LANG = "te"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        moisture = request.form.get('moisture', 'N/A')

        if 'image' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image uploaded"
            }), 400

        img_file = request.files['image']

        img = Image.open(
            io.BytesIO(img_file.read())
        ).convert('RGB')

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ---------- PREDICT ----------
        prediction = model.predict(img_array, verbose=0)

        predicted_index = int(np.argmax(prediction))
        predicted_class = class_labels[predicted_index]
        confidence = round(
            float(np.max(prediction)) * 100, 2
        )

        # ---------- CONFIDENCE CHECK ----------
        if confidence < 70:
            message = (
                "Prediction is uncertain. Please upload "
                "a clear image of a turmeric leaf."
            )

            translated_message = translator.translate(
                message, dest=TARGET_LANG
            ).text

            send_telegram_message(translated_message)

            return jsonify({
                "status": "rejected",
                "message": "Please upload a clear turmeric leaf image.",
                "confidence": confidence
            }), 200

        # ---------- FERTILIZER RECOMMENDATION ----------
        match = fertilizer_df[
            fertilizer_df['Disease_Name'] == predicted_class
        ]

        fertilizer = (
            match['Fertilizer'].values[0]
            if not match.empty
            else "No data available"
        )

        # ---------- TELEGRAM MESSAGE ----------
        message = (
            f"Disease Detected: {predicted_class}\n"
            f"Confidence: {confidence}%\n"
            f"Fertilizer: {fertilizer}\n"
            f"Soil Moisture: {moisture}%"
        )

        translated_message = translator.translate(
            message, dest=TARGET_LANG
        ).text

        send_telegram_message(translated_message)

        return jsonify({
            "status": "success",
            "disease": predicted_class,
            "confidence": confidence,
            "moisture": moisture
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


def send_telegram_message(message):
    if not BOT_TOKEN:
        raise ValueError(
            "BOT_TOKEN is missing from environment variables"
        )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=15
    )

    response.raise_for_status()


@app.route('/', methods=['GET'])
def home():
    return "Crop Disease Detection Server is Running"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
