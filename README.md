# Food Image Classifier (Food-101)

This is a Flask web app that classifies food images into one of the Food-101 categories using a TensorFlow/Keras model.

Features
- Upload any food image and get a predicted class with confidence
- Prediction alert auto-dismisses after 10 seconds
- Random food fun-facts while you wait

How it works
1. The app loads a trained model from `bbest_model1.keras` and class names from `names.json`.
2. When you upload an image, it is saved to `uploads/` and preprocessed to 224×224 RGB.
3. The model predicts probabilities; the top class, confidence, and a friendly message are rendered on `/`.

Tech stack
- Flask (serving and templating)
- TensorFlow / Keras (model inference)
- Bootstrap 5 (styling)

Project structure (key files)
- `app.py`: Flask app, image preprocessing, inference, and routing
- `templates/index.html`: UI for upload and results
- `static/`: images, styles, and assets
- `bbest_model1.keras`: trained model file (required)
- `names.json`: Food-101 class name mapping (required)

Setup
Prereqs: **Python  (tested with version 3.9) and Git.**
       : Make sure to have conda installed

1) Clone and install
```bash
git clone <this-repo-url>
cd FoodApp
conda create -n foodapp python=3.9
conda activate foodapp
pip install -r requirements.txt
```

2) Ensure model and labels exist
- Place `bbest_model1.keras` in the project root.
- Ensure `names.json` (mapping of class indices to names) is in the project root.

Run the app
```bash
python app.py
# App runs on http://0.0.0.0:7860 (accessible at http://localhost:7860)
```

Docker (optional)
If you prefer containers, a `Dockerfile` and `compose.yaml` are included.
```bash
# Build
docker build -t food101-app .
# Run
docker run -p 7860:7860 food101-app
```

Usage
1. Open `http://localhost:7860` in your browser.
2. Click “Upload a Food Image”, select a photo.
3. Click “Classify Image”. A green alert shows the prediction and confidence, then fades after ~10s.

Dataset: Food-101
-  [Overview:](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
-  [Classes list (101 categories)](https://data.vision.ee.ethz.ch/cvl/food-101/meta/classes.txt)
- [Alternative mirror (Kaggle):](https://www.kaggle.com/datasets/dansbecker/food-101)

Example dishes (learn more)
- Paella:  [https://en.wikipedia.org/wiki/Paella]
- Crème brûlée: [https://en.wikipedia.org/wiki/Cr%C3%A8me_br%C3%BBl%C3%A9e]
- Bibimbap: [https://en.wikipedia.org/wiki/Bibimbap]
- Sashimi: [https://en.wikipedia.org/wiki/Sashimi]
- Margherita pizza: [https://en.wikipedia.org/wiki/Pizza_Margherita]

Notes
- Uploaded files are stored under `uploads/`. You can clean this folder periodically.
- The app uses CPU inference by default; TensorFlow GPU support requires additional setup.
- Modify the auto-dismiss timeout in `templates/index.html` by changing the `setTimeout(..., 10000)` value.

Troubleshooting
- Model not found: Ensure `bbest_model1.keras` exists at the project root.
- Names mapping error: Ensure `names.json` is valid JSON and maps indices to class names.
- Large images: Very large files may slow uploads; consider compressing before upload.

License
This project is for educational use. Dataset licensing belongs to the Food-101 authors. See the dataset page for terms.

