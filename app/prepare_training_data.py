"""
Then we take the sentiment score and format it for training
"""

import pandas as pd
import json

# Load the CSV containing tweets and their predicted sentiment
data = pd.read_csv('classified_sentiment_text_with_confidence.csv')

# Open the output JSONL file
with open('chat_fine_tuning_dataset.jsonl', 'w', encoding='utf-8') as outfile:
    for _, row in data.iterrows():
        # Create the chat message structure
        messages = [
            {"role": "system", "content": "The assistant classifies tweets as Positive or Negative."},
            {"role": "user", "content": row['text']},
            {"role": "assistant", "content": row['predicted_sentiment']}
        ]
        
        # Write each conversation as a JSON object
        json_obj = {"messages": messages}
        outfile.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print("Chat-based fine-tuning dataset (JSONL) created successfully.")