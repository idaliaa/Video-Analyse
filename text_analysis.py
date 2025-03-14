import os
from transformers import pipeline
import pandas as pd

# Path to your EXTRACTED_AUDIO folder where the .txt transcriptions are stored
transcription_folder = "/Users/psylviana/Downloads/VideoAnalyse/EXTRACTED_AUDIO"

# Initialize Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Create a list to store the results
results = []

# Loop through each broadcaster folder and their transcriptions (text files)
for broadcaster in os.listdir(transcription_folder):
    broadcaster_path = os.path.join(transcription_folder, broadcaster)
    
    if not os.path.isdir(broadcaster_path):
        continue  # Skip if not a folder
    
    # Loop through each .txt transcription file in the broadcaster's folder
    for transcription_file in os.listdir(broadcaster_path):
        if not transcription_file.lower().endswith(".txt"):  # Process .txt files
            continue
        
        transcription_file_path = os.path.join(broadcaster_path, transcription_file)

        try:
            # Read the transcription text from the file
            with open(transcription_file_path, "r", encoding="utf-8") as file:
                transcription = file.read()

            # Perform sentiment analysis on the transcription text
            sentiment_result = sentiment_pipeline(transcription)

            # Extract sentiment label and score
            sentiment_label = sentiment_result[0]['label']
            sentiment_score = sentiment_result[0]['score']

            # Append result to the list
            results.append({
                "Broadcaster": broadcaster,
                "Video": transcription_file,
                "Sentiment": sentiment_label,
                "Score": sentiment_score
            })

        except Exception as e:
            print(f"Error processing {transcription_file}: {e}")
            continue

# Convert results to a pandas DataFrame for easier handling and saving
df = pd.DataFrame(results)

# Save results to a CSV file (optional)
df.to_csv("/Users/psylviana/Downloads/VideoAnalyse/YOLO-Video-Analysis/sentiment_analysis_results.csv", index=False)

# Print the results
print(df)
