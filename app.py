import os
import joblib
import numpy as np
import pandas as pd
import string
from flask import Flask, render_template,url_for,redirect,request,jsonify
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import traceback

app = Flask(__name__)
CORS(app)

# Load models outside the request loop to improve performance
try:
    best_model = joblib.load('best_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    print('Models Loaded Successfully.....')
except OSError as e:
    print(f'An error occurred while loading models: {e}')
    best_model, vectorizer, st_model = None, None, None

# Text Preprocessing functions
def lowercasing(txt):
    return txt.lower()

def remove_punctuations(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(txt):
    return "".join([i for i in txt if not i.isdigit()])

def remove_stopwords(txt):
    stop_words = set(stopwords.words('english'))
    try:
        words = word_tokenize(txt)
    except TypeError as e:
        print(f'Error occurred while removing stopwords: {e}')
        return ""
    
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

def remove_emojis(txt):
    return txt.encode('ascii', 'ignore').decode('ascii')

# Feature Engineering
def get_semantic_similarity(question, option):
    if st_model is None:
        return 0.0
    
    try:
        embeddings = st_model.encode([question, option])
        return cosine_similarity([embeddings[0], embeddings[1]])[0][0]
    except Exception as e:
        print('Error calculating cosine similarity:', e)
        # It's better to return a default value here so the app doesn't crash
        return 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("--- Received a request to /predict ---")
    
    if best_model is None or vectorizer is None or st_model is None:
        print("Model loading failed, returning 500.")
        return jsonify({'error': 'Models not loaded. Internal Server Error.'}), 500

    try:
        data = request.get_json()
        print(f"Received JSON data: {data}")
        
        question = data.get('question', '')
        options = data.get('options', [])

        if not question or not options or len(options) == 0:
            print("Invalid input, returning 400.")
            return jsonify({'error': 'Invalid Input. Please provide a proper question and options.'}), 400
        
        predictions = []
        for option_text in options:
            # CORRECTED: Chain the pre-processing functions together
            preprocessed_option = lowercasing(option_text)
            preprocessed_option = remove_emojis(preprocessed_option)
            preprocessed_option = remove_numbers(preprocessed_option)
            preprocessed_option = remove_punctuations(preprocessed_option)
            preprocessed_option = remove_stopwords(preprocessed_option)

            combined_text = question + " " + preprocessed_option
            tf_idf_features = vectorizer.transform([combined_text])

            semantic_score = get_semantic_similarity(question, option_text)
            semantic_features = np.array([semantic_score]).reshape(1, -1)

            combined_features = hstack([tf_idf_features, semantic_features]).tocsr()
            
            # Use `best_model.predict` to get the final class and `predict_proba` for the score
            prediction_label = int(best_model.predict(combined_features)[0])
            prediction_prob = best_model.predict_proba(combined_features)[0][1]

            predictions.append({
                'option': option_text,
                'score': prediction_prob,
                'is_correct_pred': prediction_label
            })

        # Find the best option based on the highest score
        best_option = max(predictions, key=lambda p: p['score'])

        print("Prediction successful, returning result.")
        return jsonify({
            'predicted_correct_option': best_option['option'],
            'all_predictions': predictions
        })

    except Exception as e:
        # This will now print the full error traceback in your terminal
        traceback.print_exc()
        print('There was an error during prediction:', e)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Add a message to confirm the server is running and what port to use
    print("Server is starting. Make sure your front-end points to http://127.0.0.1:5000/predict")
    app.run(debug=True, host='127.0.0.1', port=5000)
