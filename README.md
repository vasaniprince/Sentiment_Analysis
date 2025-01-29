# IMDB Sentiment Analysis Pipeline

A complete end-to-end sentiment analysis pipeline that processes IMDB movie reviews, trains a classification model, and serves predictions via a Flask API.

## Features

- Data collection from Hugging Face's IMDB dataset
- SQLite database storage for reviews
- Text preprocessing and cleaning
- Sentiment classification using Logistic Regression with TF-IDF
- Flask API for serving predictions

## Prerequisites

- Python 3.8+
- Git
- SQLite3

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sentiment-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
sentiment-analysis/
├── data_setup.py         # Database and data preparation script
├── train_model.py        # Model training script
├── app.py               # Flask application
├── imdb_reviews.db      # SQLite database
├── model.pkl            # Trained model
├── vectorizer.pkl       # TF-IDF vectorizer
├── requirements.txt
└── README.md
```

## Database Setup and Data Loading

1. Run the data setup script:
```bash
python data_setup.py
```

This script will:
- Load the IMDB dataset from Hugging Face
- Clean the review texts (lowercase, remove HTML tags, remove punctuation)
- Create an SQLite database (`imdb_reviews.db`)
- Store all reviews in the database

The database schema:
```sql
CREATE TABLE imdb_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_text TEXT,
    sentiment TEXT
);
```

## Model Training

1. Train the sentiment analysis model:
```bash
python train_model.py
```

This script will:
- Load data from the SQLite database
- Split into training and testing sets
- Convert text to TF-IDF features
- Train a Logistic Regression model
- Evaluate model performance
- Save the model and vectorizer to disk

## Model Performance

The model achieves the following metrics on the test set:
- Accuracy: 89.05%
- Precision: 88.05%
- Recall: 90.07%
- F1 Score: 89.05%

Word Cloud For Positive Review
<p align="center">
  <img src="Screenshot 2025-01-29 220756.png" alt="Accuracy Graph" width="500"/>
</p>

## Usage

### Running the Flask API

1. Start the Flask server:
```bash
python app.py
```

2. The API will be available at `http://localhost:5000`

### API Endpoint

- **Sentiment Prediction**
  - URL: `/predict`
  - Method: `POST`
  - Request Body:
    ```json
    {
        "review_text": "Your movie review text here"
    }
    ```
  - Response:
    ```json
    {
        "sentiment": "positive"  // or "negative"
    }
    ```

### Example API Request

Using curl:
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"review_text": "This movie was fantastic! I really enjoyed it."}'
```

Using Python requests:
```python
import requests

response = requests.post('http://localhost:5000/predict', 
                        json={'review_text': 'This movie was fantastic! I really enjoyed it.'})
print(response.json())
```

## Dependencies

Main dependencies include:
- pandas
- scikit-learn
- datasets (Hugging Face)
- Flask
- sqlite3
- numpy

Full list available in `requirements.txt`


## License

This project is licensed under the MIT License - see the LICENSE file for details.