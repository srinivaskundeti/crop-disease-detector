from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from googletrans import Translator
import requests
from PIL import Image
import io

app = Flask(__name__)

# ---------- LOAD MODEL AND DATA ----------
model = load_model('plant_disease_model.h5')
fertilizer_df = pd.read_csv('fertilizer_data.csv')

class_labels = ['Aphids_Disease', 'Blotch', 'Healthy_Leaf', 'Leaf_Spot']

translator = Translator()

# ---------- TELEGRAM CONFIG ----------
BOT_TOKEN = "8951131123:AAH_By6fSGJDQBuPeQ6URcW_tS19m9PZsfg"
CHAT_ID = "5982626821"
TARGET_LANG = "te"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        moisture = request.form.get('moisture', 'N/A')
        img_file = request.files['image']
        img = Image.open(io.BytesIO(img_file.read())).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = round(float(np.max(prediction)) * 100, 2)

        match = fertilizer_df[fertilizer_df['Disease_Name'] == predicted_class]
        fertilizer = match['Fertilizer'].values[0] if not match.empty else "No data available"

        message = (
            f"Disease Detected: {predicted_class}\n"
            f"Confidence: {confidence}%\n"
            f"Fertilizer: {fertilizer}\n"
            f"Soil Moisture: {moisture}%"
        )

        translated_message = translator.translate(message, dest=TARGET_LANG).text
        send_telegram_message(translated_message)

        return jsonify({"status": "success", "disease": predicted_class, "moisture": moisture})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": message})


@app.route('/', methods=['GET'])
def home():
    return "Crop Disease Detection Server is Running"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
