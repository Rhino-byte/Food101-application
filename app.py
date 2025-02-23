import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
# from tensorflow.keras.backend import clear_session

# Load the existing trained model
model = tf.keras.models.load_model("bbest_model1.keras")
model.make_predict_function() 

# Load the Food101 dataset class names
with open('names.json', 'r') as file:
    class_names = json.load(file)



# Create Flask app
app = Flask(__name__)

# Define the image size
img_shape = 224

def preprocess_image_for_inference(img_path, scale=True):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img,channels = 3)
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        return img / 255.
    else:
        return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            if "file" not in request.files:
                error = "No file uploaded"
                return render_template("index.html", error=error)

            file = request.files["file"]
            if file.filename == "":
                error = "No selected file"
                return render_template("index.html", error=error)

            file_path = os.path.join("uploads", file.filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(file_path)

            # Debugging: Check if file is uploaded correctly
            print("File uploaded:", file.filename)
            print("File path:", file_path)

            # Preprocess and predict using the CNN model
            if file and file_path:
                print("File and file path are valid.")  # New Debug Log

                img_array = preprocess_image_for_inference(file_path,scale= False)
                print("Image array shape:", img_array.shape)  # Debugging

                if img_array is not None:
                    
                    print("Image preprocessing successful.")
                    pred = model.predict(tf.expand_dims(img_array, axis=0))
                    # print("Model prediction:", prediction)  # Debugging
                    prob_pred = tf.round(float(pred.max()))
                    pred_class = class_names[str(int(pred.argmax()))]

                    

                    # Combine prediction class and probability with custom text
                    predictions = f"The model predicts this is '{pred_class}' with a confidence of {prob_pred * 100:.2f}%."


                    return render_template("index.html", prediction=predictions,error= error)
                else:
                    error = "Image preprocessing failed"
                    print(error)
            else:
                error = "Invalid file or file path"
                print(error)

        except Exception as e:
            error = str(e)
            print("Error occurred:", error)
            return render_template("index.html", error=error)

    return render_template("index.html", prediction=None, error=None)

if __name__ == "__main__":
    app.run(debug=True)
