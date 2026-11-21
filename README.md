
# Communication and Intent Detection

## Project Overview
This project uses Natural Language Processing (NLP) and Machine Learning to identify the intent behind a user's text message.

## Technologies Used
- Python
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- LinearSVC
- Flask
- HTML, CSS and JavaScript
- NLTK and scikit-learn

## How It Works
1. The user enters a message.
2. The application preprocesses the text.
3. TF-IDF converts the text into numerical features.
4. A trained LinearSVC model predicts the intent.
5. The Flask application returns the result to the frontend.

## Features
- Text-based intent prediction
- Web interface for entering messages
- Confidence estimate
- Low-confidence / unknown intent handling

## How to Run Locally
1. Install Python.
2. Install dependencies using `pip install -r requirements.txt`.
3. Ensure the trained model and TF-IDF vectorizer files are available.
4. Run `python app.py`.
5. Open `http://127.0.0.1:5000` in your browser.

## Note
The displayed confidence is an estimate derived from model decision scores; it is not a calibrated probability.