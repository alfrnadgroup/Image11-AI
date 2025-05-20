from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model once when the app starts
model = tf.keras.models.load_model("model/animal_model.h5")

# This should match the classes your model was trained on
class_labels = ['animal', 'bird', 'tree']  # Replace with your actual class order!

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        if "file" not in request.files:
            prediction = "No file part"
            return render_template("index.html", prediction=prediction)

        file = request.files["file"]
        if file.filename == "":
            prediction = "No selected file"
            return render_template("index.html", prediction=prediction)

        if file:
            # Save uploaded file temporarily
            img_path = os.path.join("uploads", file.filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(img_path)

            # Load and preprocess the image
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array)
            predicted_index = np.argmax(preds, axis=1)[0]
            predicted_label = class_labels[predicted_index]

            prediction = f"Prediction: {predicted_label}"

            # Optionally delete the uploaded file after prediction
            os.remove(img_path)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
