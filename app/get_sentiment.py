"""
First, we get the sentiment scores
"""
from transformers import pipeline
import pandas as pd

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Load tweets from CSV
sentiment_text_df = pd.read_csv('sentiment_text.csv')

def classify_tweet(text):
    result = sentiment_pipeline(text)[0]  # Get the classification result
    label = result['label']  # POSITIVE or NEGATIVE
    score = result['score']  # Confidence score

    # Return sentiment and flag low-confidence tweets
    return label, score, "Review" if score < 0.8 else "Auto-Approved"

# Classify tweets and store results in new columns
sentiment_text_df[['predicted_sentiment', 'confidence_score', 'review_status']] = sentiment_text_df['text'].apply(
    lambda text: pd.Series(classify_tweet(text))
)

# Save classified tweets to a new CSV
sentiment_text_df.to_csv('classified_sentiment_text_with_confidence.csv', index=False)

print("Classification complete. Low-confidence tweets are flagged for review.")