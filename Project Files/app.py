import os
import numpy as np
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load trained model
model = load_model("dogbreed.h5")

# Upload folder
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class labels
class_names = sorted(os.listdir("dataset/train"))

# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICT PAGE ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        if "file" not in request.files:
            return "No file uploaded"

        file = request.files["file"]

        if file.filename == "":
            return "No selected file"

        # Save image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]

        image_url = url_for('static', filename='uploads/' + file.filename)

        return render_template(
            "output.html",
            prediction=predicted_class,
            image_path=image_url
        )

    return render_template("predict.html")


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
