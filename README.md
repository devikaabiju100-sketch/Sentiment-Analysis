# 🤖 Sentiment Analysis Project

A complete machine learning application that analyzes text sentiment in real-time using a trained classification model and provides a beautiful web interface.

## ✨ Features

- **AI-Powered Sentiment Detection**: Classifies text as Positive, Negative, or Neutral
- **Machine Learning Model**: Uses TF-IDF vectorization with Multinomial Naive Bayes
- **Web Interface**: Modern, responsive Flask web application
- **Real-time Analysis**: Instant sentiment predictions with confidence scores
- **Single-File Architecture**: All functionality in one easy-to-run Python file
- **Auto-Training**: Automatically trains on sample data if no model exists

## 🛠️ Tech Stack

- **Backend**: Flask
- **ML Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML5, CSS3
- **Language**: Python 3.8+

## 📋 Requirements

- Python 3.8 or higher
- pip (Python package manager)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/devikaabiju100-sketch/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python sentiment_analysis_complete.py
```

### 4. Open in Browser
Navigate to: `http://localhost:5000`

## 📖 Usage

1. Enter any text in the textarea (product review, comment, feedback, etc.)
2. Click "Analyze Sentiment"
3. View the predicted sentiment with confidence score
4. Emoji indicators show the sentiment at a glance:
   - 😊 Positive
   - 😞 Negative
   - 😐 Neutral

## 📁 Project Structure

```
Sentiment-Analysis/
├── sentiment_analysis_complete.py  # Main application file
├── requirements.txt                # Project dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── sentiment_model.pkl             # Trained model (auto-generated)
└── vectorizer.pkl                  # TF-IDF vectorizer (auto-generated)
```

## 🎯 How It Works

### Model Training
- Initializes with 20 sample reviews for demonstration
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Trains a Multinomial Naive Bayes classifier
- Achieves ~95% accuracy on training data
- Automatically saves trained model and vectorizer

### Prediction Pipeline
1. User input text is received via web form
2. Text is vectorized using the trained TF-IDF vectorizer
3. Model predicts sentiment class (Positive/Negative/Neutral)
4. Confidence score is calculated from prediction probabilities
5. Results are displayed with visual indicators

## 📊 Sample Training Data

The model includes sample reviews for demonstration:
- ✅ Positive reviews (e.g., "This movie is absolutely fantastic!")
- ❌ Negative reviews (e.g., "Terrible film, waste of time.")
- 😐 Neutral reviews (e.g., "It was okay, nothing special.")

## 🔧 Customization

### Train with Your Own Dataset

Modify the `train_model()` function in `sentiment_analysis_complete.py`:

```python
def train_model(sample_reviews=None):
    if sample_reviews is None:
        # Load your CSV file
        data = pd.read_csv("path/to/your/dataset.csv")
    # ... rest of training code
```

Your CSV should have columns: `review` and `sentiment`

### Adjust Model Parameters

In the `train_model()` function, you can modify:
- `max_features`: Number of features for TF-IDF (default: 5000)
- `ngram_range`: Word n-grams to use (default: (1, 2))
- `test_size`: Train-test split ratio (default: 0.2)

## 📈 Model Performance

```
Model Accuracy: ~95% (on sample data)
Training Time: < 1 second
Prediction Time: < 100ms per review
```

## 🎨 UI Features

- **Gradient Background**: Purple gradient color scheme
- **Responsive Design**: Works on desktop and mobile
- **Real-time Feedback**: Instant visual feedback on analysis
- **Smooth Animations**: Button hover effects and transitions
- **Clear Results**: Large, easy-to-read confidence scores

## 🔐 Security

- Input validation on all text submissions
- No external API calls or data transmission
- All processing happens locally
- No user data is stored or logged

## 🐛 Troubleshooting

### Port Already in Use
If port 5000 is in use, modify the last line:
```python
app.run(debug=True, port=5001)  # Use different port
```

### Model Not Loading
Delete the pickle files and restart:
```bash
rm sentiment_model.pkl vectorizer.pkl
python sentiment_analysis_complete.py
```

### Dependencies Error
Reinstall all requirements:
```bash
pip install --upgrade -r requirements.txt
```

## 📚 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created by [devikaabiju100-sketch](https://github.com/devikaabiju100-sketch)

## 🙌 Acknowledgments
- Consolidated into single-file architecture for easier deployment
- Built with ❤️ for sentiment analysis enthusiasts

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ | [View on GitHub](https://github.com/devikaabiju100-sketch/Sentiment-Analysis)**
