from flask import Flask, request, render_template, jsonify
import pickle
import re

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to clean text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve data from the form
        review_text = request.form['review_text']
        
        # Validate input
        if not review_text.strip():
            return jsonify({'error': 'Please enter a valid review.'}), 400
        
        # Clean the text
        cleaned_text = clean_text(review_text)

        # Convert to TF-IDF vector
        text_tfidf = vectorizer.transform([cleaned_text])

        # Predict sentiment
        prediction = model.predict(text_tfidf)
        # print(prediction)
        sentiment = 'positive' if prediction[0] == 1 else 'negative'

        # Render the result in the same HTML template
        return render_template('index.html', prediction=f'Sentiment: {sentiment}')
    
    return jsonify({'error': 'Invalid request method.'}), 405

if __name__ == '__main__':
    app.run(debug=True)