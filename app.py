from flask import Flask, request, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

app = Flask(__name__, template_folder='templates')

# Load model huruf/aksara Pegon
model = load_model("base_model.h5")

# Daftar label kelas (ubah sesuai urutan training)
class_names = ["C", "G", "Ng", "Ny", "P"]

# Threshold minimum probabilitas
CONF_THRESHOLD = 0.7  # 70%

def predict_character(img_path):
    # Load gambar grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Resize ke 224x224
    img = cv2.resize(img, (224, 224))

    # Normalisasi
    img = img / 255.0

    # Expand dims → (1, 224, 224, 1)
    img_array = np.expand_dims(img, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    confidence = float(np.max(predictions))  # antara 0–1

    # Jika confidence < threshold → anggap bukan huruf Pegon
    if confidence < CONF_THRESHOLD:
        return "Bukan huruf Pegon", round(confidence * 100, 2)

    return class_names[predicted_class], round(confidence * 100, 2)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    img_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        if file:
            # Simpan gambar upload
            static_folder = os.path.join(app.root_path, 'static')
            os.makedirs(static_folder, exist_ok=True)
            filepath = os.path.join(static_folder, 'uploaded_image.png')
            file.save(filepath)

            # Prediksi huruf/aksara Pegon
            prediction, confidence = predict_character(filepath)
            img_path = 'static/uploaded_image.png'
            img_path = img_path.replace('\\', '/')

    return render_template('index.html', prediction=prediction, confidence=confidence, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
