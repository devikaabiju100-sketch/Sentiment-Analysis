"""
Sentiment Analysis System - Complete Application
Combines model training, vectorization, and Flask web interface
"""

import pandas as pd
import pickle
import os
from io import StringIO
from flask import Flask, render_template_string, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ============================================================================
# SENTIMENT ANALYSIS MODEL TRAINING
# ============================================================================

class SentimentModel:
    """Train and manage sentiment analysis model"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.accuracy = None
    
    def create_sample_dataset(self):
        """Create a sample dataset if real IMDB dataset is not available"""
        sample_reviews = [
            ("This movie was absolutely fantastic! I loved every minute of it.", "positive"),
            ("One of the best films I've ever seen. Brilliant acting and storytelling.", "positive"),
            ("Amazing cinematography and great plot. Highly recommended!", "positive"),
            ("I couldn't stop watching. Simply wonderful!", "positive"),
            ("This is a masterpiece. Best movie of the year!", "positive"),
            ("Great movie with excellent performances.", "positive"),
            ("Loved it! Would watch again.", "positive"),
            ("Absolutely terrible. Waste of time.", "negative"),
            ("This movie was boring and poorly made.", "negative"),
            ("Worst film I've ever seen. Don't watch!", "negative"),
            ("Awful plot and terrible acting. Disappointing.", "negative"),
            ("Complete disaster. Not recommended at all.", "negative"),
            ("Really bad. Couldn't finish watching it.", "negative"),
            ("Disappointing and dull. Not worth watching.", "negative"),
            ("Okay movie, nothing special but decent.", "positive"),
            ("Not bad but could be better.", "positive"),
            ("Interesting plot with some good scenes.", "positive"),
            ("Average film. Some good moments.", "positive"),
        ]
        
        # Expand dataset with variations
        expanded_reviews = sample_reviews * 50  # Repeat to get more data
        
        data = pd.DataFrame(expanded_reviews, columns=['review', 'sentiment'])
        return data
    
    def train(self, data=None):
        """Train the sentiment model"""
        print("=" * 60)
        print("SENTIMENT ANALYSIS MODEL TRAINING")
        print("=" * 60)
        
        # Use provided data or create sample
        if data is None:
            print("Creating sample dataset...")
            data = self.create_sample_dataset()
        else:
            print("Using provided dataset...")
        
        print(f"Dataset size: {len(data)} reviews")
        print(f"Sentiment distribution:\n{data['sentiment'].value_counts()}\n")
        
        # Prepare features and labels
        X = data['review']
        y = data['sentiment']
        
        # Capitalize labels for consistency
        y = y.str.capitalize()
        
        # Split data
        print("Splitting data into train/test sets (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Vectorization
        print("Creating TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            min_df=1,
            max_df=0.9
        )
        
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        print(f"Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
        
        # Train model
        print("\nTraining Multinomial Naive Bayes model...")
        self.model = MultinomialNB()
        self.model.fit(X_train_vectorized, y_train)
        
        # Evaluate
        print("Evaluating model...")
        predictions = self.model.predict(X_test_vectorized)
        self.accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n{'=' * 60}")
        print(f"Model Accuracy: {self.accuracy * 100:.2f}%")
        print(f"{'=' * 60}\n")
        
        return self.accuracy
    
    def predict(self, text):
        """Predict sentiment of given text"""
        if self.model is None or self.vectorizer is None:
            return None, None
        
        transformed_text = self.vectorizer.transform([text])
        sentiment = self.model.predict(transformed_text)[0]
        confidence = max(self.model.predict_proba(transformed_text)[0]) * 100
        
        return sentiment, confidence


# ============================================================================
# FLASK WEB APPLICATION
# ============================================================================

# Initialize Flask app and model
app = Flask(__name__)
sentiment_analyzer = SentimentModel()

# Train model on startup
sentiment_analyzer.train()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 AI Sentiment Analysis System</title>
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
            max-width: 800px;
            width: 100%;
            padding: 45px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            animation: slideIn 0.5s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .subtitle {
            color: #666;
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        
        .accuracy-badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-top: 10px;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 10px;
            color: #333;
            font-weight: 600;
            font-size: 1.05em;
        }
        
        textarea {
            width: 100%;
            height: 180px;
            padding: 20px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 1em;
            font-family: inherit;
            resize: vertical;
            transition: all 0.3s ease;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .results {
            margin-top: 40px;
            padding-top: 40px;
            border-top: 2px solid #e0e0e0;
            animation: fadeIn 0.5s ease-out;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        
        .result-item {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .sentiment-result {
            font-size: 2.5em;
            font-weight: bold;
            margin: 20px 0;
            padding: 20px;
            border-radius: 12px;
        }
        
        .sentiment-positive {
            background: #d4edda;
            color: #155724;
        }
        
        .sentiment-negative {
            background: #f8d7da;
            color: #721c24;
        }
        
        .sentiment-neutral {
            background: #fff3cd;
            color: #856404;
        }
        
        .confidence-meter {
            margin: 20px 0;
        }
        
        .confidence-label {
            color: #666;
            font-size: 1.05em;
            margin-bottom: 10px;
        }
        
        .confidence-bar {
            width: 100%;
            height: 25px;
            background: #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
            transition: width 0.5s ease;
        }
        
        .footer {
            margin-top: 40px;
            text-align: center;
            color: #999;
            font-size: 0.9em;
            border-top: 1px solid #e0e0e0;
            padding-top: 20px;
        }
        
        .info-box {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 4px solid #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Sentiment Analysis</h1>
            <p class="subtitle">Analyze the sentiment of your text</p>
            <div class="accuracy-badge">
                ✓ Model Accuracy: {{ accuracy }}%
            </div>
        </div>
        
        <form method="POST">
            <div class="form-group">
                <label for="review">Enter your text or review:</label>
                <textarea 
                    id="review"
                    name="review" 
                    placeholder="Type your review, comment, or any text here..." 
                    required></textarea>
            </div>
            
            <button type="submit">🔍 Analyze Sentiment</button>
        </form>
        
        {% if prediction %}
        <div class="results">
            <div class="result-item">
                <div class="sentiment-result {{ sentiment_class }}">
                    {{ prediction }}
                </div>
                
                <div class="confidence-meter">
                    <div class="confidence-label">
                        Confidence Level
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {{ confidence_value }}%">
                            {{ confidence }}
                        </div>
                    </div>
                </div>
                
                <div class="info-box">
                    <strong>Analysis:</strong> The system predicts this text has a 
                    <strong>{{ prediction }}</strong> sentiment with 
                    <strong>{{ confidence }}</strong> confidence.
                </div>
            </div>
        </div>
        {% endif %}
        
        <div class="footer">
            <p>🚀 Machine Learning Project | Flask + Scikit-Learn</p>
            <p>Powered by TF-IDF Vectorization & Multinomial Naive Bayes</p>
        </div>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    """Main application route"""
    prediction = ""
    confidence = ""
    confidence_value = 0
    sentiment_class = ""
    
    if request.method == "POST":
        review = request.form.get("review", "").strip()
        
        if review:
            sentiment, conf = sentiment_analyzer.predict(review)
            
            if sentiment:
                # Format prediction
                if sentiment == "Positive":
                    prediction = "😊 Positive Sentiment"
                    sentiment_class = "sentiment-positive"
                elif sentiment == "Negative":
                    prediction = "😞 Negative Sentiment"
                    sentiment_class = "sentiment-negative"
                else:
                    prediction = "😐 Neutral Sentiment"
                    sentiment_class = "sentiment-neutral"
                
                confidence = f"{conf:.2f}%"
                confidence_value = conf
    
    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        confidence=confidence,
        confidence_value=confidence_value,
        sentiment_class=sentiment_class,
        accuracy=f"{sentiment_analyzer.accuracy * 100:.2f}" if sentiment_analyzer.accuracy else "N/A"
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS APPLICATION STARTED")
    print("=" * 60)
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
