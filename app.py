
import os
import io
import requests
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from googletrans import Translator
from PIL import Image, UnidentifiedImageError

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

# ---------- PREDICTION SETTINGS ----------
CONFIDENCE_THRESHOLD = 70.0


def reject_image(reason, confidence=None):
    """Reject an image and notify the farmer in Telugu."""
    message = (
        "దయచేసి పసుపు ఆకును స్పష్టంగా ఫోటో తీసి "
        "మళ్లీ పంపండి. చేతులు లేదా ఇతర వస్తువులను "
        "కాకుండా ఆకును మాత్రమే చూపించండి."
    )

    try:
        send_telegram_message(message)
    except Exception as telegram_error:
        app.logger.error(
            "Telegram notification failed: %s",
            telegram_error
        )

    result = {
        "status": "rejected",
        "message": reason
    }

    if confidence is not None:
        result["confidence"] = confidence

    return jsonify(result), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        moisture = request.form.get('moisture', 'N/A')

        # ---------- CHECK IMAGE UPLOAD ----------
        if 'image' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image uploaded"
            }), 400

        img_file = request.files['image']

        if not img_file.filename:
            return jsonify({
                "status": "error",
                "message": "Please select an image"
            }), 400

        # ---------- READ IMAGE ----------
        try:
            img = Image.open(
                io.BytesIO(img_file.read())
            ).convert('RGB')
        except (UnidentifiedImageError, OSError, ValueError):
            return jsonify({
                "status": "error",
                "message": "Invalid image. Please upload a valid image."
            }), 400

        # ---------- BASIC IMAGE QUALITY CHECK ----------
        img_array_original = np.array(img)

        brightness = float(np.mean(img_array_original))

        if brightness < 15 or brightness > 245:
            return reject_image(
                "Image is too dark or too bright. "
                "Please upload a clear turmeric leaf image."
            )

        # ---------- PREPROCESS IMAGE ----------
        img = img.resize((224, 224))

        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ---------- PREDICT ----------
        prediction = model.predict(img_array, verbose=0)
        prediction = np.asarray(prediction)

        # ---------- VALIDATE MODEL OUTPUT ----------
        if (
            prediction.ndim != 2
            or prediction.shape[0] != 1
            or prediction.shape[1] != len(class_labels)
            or not np.all(np.isfinite(prediction))
        ):
            return jsonify({
                "status": "error",
                "message": "Invalid model prediction output"
            }), 500

        probabilities = prediction[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_labels[predicted_index]

        confidence = round(
            float(probabilities[predicted_index]) * 100,
            2
        )

        # ---------- CONFIDENCE CHECK ----------
        if confidence < CONFIDENCE_THRESHOLD:
            return reject_image(
                "Low confidence. Please upload a clear "
                "turmeric leaf image.",
                confidence
            )

        # ---------- FERTILIZER RECOMMENDATION ----------
        match = fertilizer_df[
            fertilizer_df['Disease_Name'] == predicted_class
        ]

        fertilizer = (
            str(match['Fertilizer'].iloc[0])
            if not match.empty
            else "No data available"
        )

        # ---------- TELEGRAM MESSAGE IN TELUGU ----------
        message = (
            f"వ్యాధి పేరు: {predicted_class}\n"
            f"నమ్మక స్థాయి: {confidence}%\n"
            f"ఉపయోగించాల్సిన ఎరువు: {fertilizer}\n"
            f"నేలలో తేమ స్థాయి: {moisture}%"
        )

        send_telegram_message(message)

        # ---------- RESPONSE TO HTML TEST PAGE ----------
        return jsonify({
            "status": "success",
            "disease": predicted_class,
            "confidence": confidence,
            "fertilizer": fertilizer,
            "moisture": moisture
        }), 200

    except Exception as e:
        app.logger.exception("Prediction failed")

        return jsonify({
            "status": "error",
            "message": "Prediction failed. Please try again."
        }), 500


def send_telegram_message(message):
    if not BOT_TOKEN:
        raise ValueError(
            "BOT_TOKEN is missing from environment variables"
        )

    url = (
        f"https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )

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
