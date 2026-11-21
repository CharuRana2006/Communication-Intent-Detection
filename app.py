from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("intent_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# NLP resources
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Confidence threshold
CONFIDENCE_THRESHOLD = 30.0

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict intent
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get message from frontend
        data = request.get_json()
        message = data.get("message", "")

        # Check empty message
        if not message.strip():
            return jsonify({
                "error": "Please enter a message."
            })

        # Preprocess message
        cleaned_message = preprocess_text(message)

        # Convert text into TF-IDF features
        message_tfidf = tfidf.transform([cleaned_message])

        # Predict intent
        prediction = model.predict(message_tfidf)[0]

        # Calculate confidence
        confidence = None

        # Models having predict_proba()
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(message_tfidf)[0]

            confidence = float(
                np.max(probabilities)
            ) * 100

        # LinearSVC
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(message_tfidf)

            scores = np.asarray(scores).flatten()

            # Convert decision scores to relative probabilities
            exp_scores = np.exp(
                scores - np.max(scores)
            )

            probabilities = (
                exp_scores / np.sum(exp_scores)
            )

            confidence = float(
                np.max(probabilities)
            ) * 100

        # Check for unknown intent
        if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
            result = {
                "intent": "unknown",
                "confidence": round(confidence, 2),
                "message": "Unable to confidently identify the intent. Please enter a more relevant message."
            }

        else:
            result = {
                "intent": str(prediction),
                "confidence": round(confidence, 2)
                if confidence is not None
                else None
            }

        return jsonify(result)

    except Exception as e:
        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500

# Run Flask application
if __name__ == "__main__":
    app.run(debug=True)