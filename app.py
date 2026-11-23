
from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
import numpy as np
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# -----------------------------------
# Load trained model and TF-IDF
# -----------------------------------

model = joblib.load("intent_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------------
# Load NLP resources
# -----------------------------------

nltk_data_path = os.environ.get(
    "NLTK_DATA",
    "/usr/local/share/nltk_data"
)

nltk.data.path.append(nltk_data_path)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------------
# Text preprocessing
# -----------------------------------

def preprocess_text(text):

    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# -----------------------------------
# Home page
# -----------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------------
# Health check
# -----------------------------------

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy",
        "message": "Communication and Intent Detection API is running"
    })


# -----------------------------------
# Predict intent
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get JSON data
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Invalid request. Please send JSON data."
            }), 400

        message = data.get("message", "")

        # Validate message
        if not isinstance(message, str) or not message.strip():

            return jsonify({
                "error": "Please enter a valid message."
            }), 400

        # Preprocess input
        cleaned_message = preprocess_text(message)

        if not cleaned_message:

            return jsonify({
                "error": "Please enter a meaningful message."
            }), 400

        # Convert text into TF-IDF features
        message_tfidf = tfidf.transform([cleaned_message])

        # Check whether input contains known TF-IDF features
        if message_tfidf.nnz == 0:

            return jsonify({
                "intent": "unknown",
                "confidence": 0,
                "score_type": "no_matching_features",
                "message": "This message does not match any known intent."
            })

        # Predict intent using saved SVM model
        prediction = model.predict(message_tfidf)[0]

        # Get SVM decision scores
        # These are NOT calibrated probabilities.
        relative_score = None

        if hasattr(model, "decision_function"):

            scores = np.asarray(
                model.decision_function(message_tfidf)
            ).flatten()

            # Relative score for display/diagnostics only
            exp_scores = np.exp(
                scores - np.max(scores)
            )

            probabilities = exp_scores / np.sum(exp_scores)

            relative_score = round(
                float(np.max(probabilities)) * 100,
                2
            )

        # Print diagnostic information
        print("--------------------------------")
        print("Original Message:", message)
        print("Cleaned Message:", cleaned_message)
        print("Matched Features:", message_tfidf.nnz)
        print("RAW PREDICTION:", prediction)
        print("Relative Score:", relative_score)
        print("--------------------------------")

        # Return predicted intent
        return jsonify({

            "intent": str(prediction),

            # This is a relative score, not true confidence.
            "confidence": relative_score,

            "score_type": "uncalibrated_relative_score",

            "message": "Intent prediction completed successfully."

        })

    except Exception as e:

        print("PREDICTION ERROR:", str(e))

        return jsonify({
            "error": "An error occurred while predicting the intent."
        }), 500


# -----------------------------------
# Run Flask application
# -----------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
