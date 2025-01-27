# data_setup.py
import sqlite3
import pandas as pd
from datasets import load_dataset
import re

# Function to clean text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Load the IMDB dataset
dataset = load_dataset("imdb")
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Combine train and test data
df = pd.concat([train_df, test_df], ignore_index=True)

# Clean the review text
df['cleaned_review_text'] = df['text'].apply(clean_text)

# Connect to SQLite database
conn = sqlite3.connect('imdb_reviews.db')
cursor = conn.cursor()

# Create a table to store the reviews
cursor.execute('''
CREATE TABLE IF NOT EXISTS imdb_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_text TEXT,
    sentiment TEXT
)
''')

# Insert data into the table
for index, row in df.iterrows():
    cursor.execute('''
    INSERT INTO imdb_reviews (review_text, sentiment)
    VALUES (?, ?)
    ''', (row['cleaned_review_text'], 'positive' if row['label'] == 1 else 'negative'))

# Commit the transaction and close the connection
conn.commit()
conn.close()

print("Data setup complete. Database 'imdb_reviews.db' created.")