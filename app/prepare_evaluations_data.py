import pandas as pd

# Load the reviewed and classified CSV
data = pd.read_csv('classified_sentiment_text_with_confidence.csv')

# # Function to map sentiment to numeric labels
# def map_sentiment_to_label(sentiment):
#     if sentiment == "NEGATIVE":
#         return 0
#     elif sentiment == "POSITIVE":
#         return 1
#     else:
#         raise ValueError(f"Unknown sentiment: {sentiment}")

# Apply the mapping function to create a label column
# data['label'] = data['predicted_sentiment'].apply(map_sentiment_to_label)
data['label'] = data['predicted_sentiment']

# Create the CSV for Evals
output_csv = 'evals_tweets_dataset.csv'
data[['text', 'label']].rename(columns={'text': 'text'}).to_csv(output_csv, index=False)

print(f"CSV for Evals created successfully: {output_csv}")

"""
Select columns with model's responses
DO NOT SELECT ANYTHING
"""

"""
Generate new responses 
CHECK THIS BOX
"""

"""
SYSTEM MESSAGE:
You are an expert in analyzing the sentiment of reviews.

- If the review is positive, you will output POSITIVE
- If the review is negative, you will output NEGATIVE
- Only output POSITIVE or NEGATIVE as a response - do not output any other text
"""

"""
USER MESSAGE:
Determine if the following review is positive or negative:
{{item.text}}
"""