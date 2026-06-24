"""
Complete Sentiment Analysis Project - Single File
Integrates model training and Flask web application
"""

import os
import pickle
import pandas as pd
from io import StringIO
from flask import Flask, render_template_string, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ============================================================================
# MODEL TRAINING SECTION
# ============================================================================

MODEL_FILE = "sentiment_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

def train_model(sample_reviews=None):
    """
    Train sentiment analysis model.
    If no data provided, uses sample reviews for demonstration.
    """
    
    if sample_reviews is None:
        # Sample reviews for demonstration
        sample_data = {
            'review': [
                'This movie is absolutely fantastic! I loved every minute.',
                'Terrible film, waste of time. Very disappointing.',
                'It was okay, nothing special but watchable.',
                'Amazing performances and brilliant storytelling!',
                'Horrible acting and boring plot. Not recommended.',
                'Not bad, had some good moments.',
                'One of the best movies I have ever seen!',
                'Awful screenplay and poor direction.',
                'Pretty good, I enjoyed it quite a bit.',
                'Mediocre at best, could have been better.',
                'Excellent film, highly entertaining!',
                'Worst movie ever made.',
                'Pleasant surprise, better than expected.',
                'Absolutely dreadful, unwatchable garbage.',
                'Great cinematography and compelling narrative.',
                'Boring and forgettable.',
                'Outstanding production quality!',
                'Really disappointing effort.',
                'Wonderful story with great acting.',
                'Painfully bad from start to finish.'
            ],
            'sentiment': [
                'positive', 'negative', 'neutral', 'positive', 'negative',
                'neutral', 'positive', 'negative', 'positive', 'neutral',
                'positive', 'negative', 'positive', 'negative', 'positive',
                'negative', 'positive', 'negative', 'positive', 'negative'
            ]
        }
        data = pd.DataFrame(sample_data)
    else:
        data = sample_reviews
    
    print("=" * 60)
    print("TRAINING SENTIMENT ANALYSIS MODEL")
    print("=" * 60)
    print(f"Total reviews: {len(data)}")
    
    # Prepare features and labels
    X = data["review"]
    y = data["sentiment"].str.capitalize()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Vectorize text
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )
    
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    
    # Evaluate
    predictions = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("=" * 60)
    
    # Save model
    pickle.dump(model, open(MODEL_FILE, "wb"))
    pickle.dump(vectorizer, open(VECTORIZER_FILE, "wb"))
    
    return model, vectorizer

# ============================================================================
# FLASK WEB APPLICATION SECTION
# ============================================================================

app = Flask(__name__)

# Load or train model
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    print("Loading existing model...")
    model = pickle.load(open(MODEL_FILE, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_FILE, "rb"))
else:
    print("Training new model...")
    model, vectorizer = train_model()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Sentiment Analysis System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            width: 100%;
            max-width: 700px;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        form {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        textarea {
            width: 100%;
            height: 150px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        button {
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .result-section {
            margin-top: 30px;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 10px;
            display: none;
        }
        
        .result-section.show {
            display: block;
        }
        
        .result {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        
        .confidence {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 10px;
        }
        
        .sentiment-positive {
            color: #27ae60;
        }
        
        .sentiment-negative {
            color: #e74c3c;
        }
        
        .sentiment-neutral {
            color: #f39c12;
        }
        
        .footer {
            margin-top: 30px;
            color: #999;
            font-size: 0.9em;
        }
        
        .info-box {
            background: #e8f4f8;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: left;
            font-size: 0.9em;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Sentiment Analysis</h1>
        <p class="subtitle">AI-powered text sentiment detection</p>
        
        <div class="info-box">
            <strong>💡 Tip:</strong> Enter any text (review, comment, feedback) and the AI will analyze its sentiment as Positive, Negative, or Neutral.
        </div>
        
        <form method="POST">
            <textarea 
                name="review" 
                placeholder="Enter your text here... (e.g., product review, feedback, comment)"
                required></textarea>
            <button type="submit">Analyze Sentiment</button>
        </form>
        
        {% if prediction %}
        <div class="result-section show">
            <div class="result">
                <span class="sentiment-{{ prediction_class }}">{{ emoji }} {{ prediction }}</span>
            </div>
            <div class="confidence">
                Confidence: <strong>{{ confidence }}</strong>
            </div>
        </div>
        {% endif %}
        
        <div class="footer">
            Machine Learning Powered by Scikit-Learn & Flask
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    """Main route for sentiment analysis"""
    
    prediction = ""
    confidence = ""
    emoji = ""
    prediction_class = ""
    
    if request.method == "POST":
        review = request.form.get("review", "").strip()
        
        if review:
            # Vectorize input
            transformed_review = vectorizer.transform([review])
            
            # Predict sentiment
            sentiment = model.predict(transformed_review)[0]
            probabilities = model.predict_proba(transformed_review)[0]
            confidence_score = max(probabilities) * 100
            
            # Format output
            if sentiment == "Positive":
                prediction = "Positive"
                emoji = "😊"
                prediction_class = "positive"
            elif sentiment == "Negative":
                prediction = "Negative"
                emoji = "😞"
                prediction_class = "negative"
            else:
                prediction = "Neutral"
                emoji = "😐"
                prediction_class = "neutral"
            
            confidence = f"{confidence_score:.2f}%"
    
    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        emoji=emoji,
        confidence=confidence,
        prediction_class=prediction_class
    )

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS APPLICATION STARTED")
    print("=" * 60)
    print("Access the application at: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True)
