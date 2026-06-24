# 🤖 Sentiment Analysis System

A complete, production-ready sentiment analysis application that combines machine learning model training with a beautiful Flask web interface.

## 📋 Features

✅ **Complete in One File** - All functionality in a single Python file
✅ **Automatic Model Training** - Trains on sample data on startup
✅ **TF-IDF Vectorization** - Advanced text feature extraction
✅ **Multinomial Naive Bayes** - Proven ML algorithm for text classification
✅ **Beautiful Web Interface** - Modern, responsive Flask UI with animations
✅ **Real-time Predictions** - Instant sentiment analysis with confidence scores
✅ **Visual Confidence Meter** - Progress bar showing prediction certainty
✅ **No External Dataset Required** - Works out of the box with sample data

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/devikaabiju100-sketch/Sentiment-Analysis.git
cd Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
python sentiment_analysis_app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

## 📊 How It Works

1. **Data Preparation**: The app creates a sample dataset with positive and negative movie reviews
2. **Text Vectorization**: Reviews are converted to numerical features using TF-IDF
3. **Model Training**: Multinomial Naive Bayes classifier is trained on the data
4. **Evaluation**: Model accuracy is calculated on test data
5. **Web Interface**: User can input text through a beautiful web interface
6. **Prediction**: The trained model predicts sentiment (Positive/Negative/Neutral) with confidence scores

## 🛠️ Tech Stack

- **Python 3.8+**
- **Flask** - Web framework
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

## 📈 Model Details

- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: 10,000 max features
- **Train/Test Split**: 80/20
- **Accuracy**: ~95% (on sample data)

## 🎨 User Interface

The web interface includes:
- Clean, modern design with gradient backgrounds
- Smooth animations and transitions
- Large textarea for text input
- Sentiment result with emoji indicators
- Visual confidence meter with percentage
- Model accuracy badge
- Responsive design for mobile devices

## 💡 Example Usage

1. Open the web interface
2. Enter a review or comment:
   - "This movie was absolutely fantastic! Loved every minute."
3. Click "Analyze Sentiment" button
4. View the prediction:
   - **Result**: 😊 Positive Sentiment
   - **Confidence**: 95.45%

## 🔧 Customization

To use your own dataset:

```python
# Modify the train() call in sentiment_analysis_app.py
data = pd.read_csv('your_dataset.csv')  # with 'review' and 'sentiment' columns
sentiment_analyzer.train(data=data)
```

## 📝 Dataset Format

If using a custom dataset, it should have two columns:
- `review`: The text content
- `sentiment`: Label (positive/negative or similar)

```csv
review,sentiment
"This movie was great!",positive
"Terrible experience.",negative
```

## 🐛 Troubleshooting

**Port already in use**:
```bash
# Change the port in sentiment_analysis_app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

**Dependencies not installing**:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 📚 Project Structure

```
Sentiment-Analysis/
├── sentiment_analysis_app.py  # Main application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🎓 Learning Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn NLP Guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Naive Bayes Classification](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [TF-IDF Vectorization](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created as a demonstration of combining Machine Learning with Web Development.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

---

**Happy Analyzing! 🚀**
