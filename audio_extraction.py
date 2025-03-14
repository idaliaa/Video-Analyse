#!/Users/psylviana/Downloads/VideoAnalyse/YOLO-Video-Analysis/yolov8_env/bin/python
import os
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to main video folder
video_folder = "/Users/psylviana/Downloads/VideoAnalyse/VIDEOS"

# Path to store the extracted audio
audio_output_folder = "/Users/psylviana/Downloads/VideoAnalyse/EXTRACTED_AUDIO"

# Create output folder if it doesn't exist
os.makedirs(audio_output_folder, exist_ok=True)

# Initialize Whisper model (choose the model size: base, small, medium, large)
model = whisper.load_model("base") 
# Loop through each broadcaster folder (e.g., ZDF, AFD, CNN)
for broadcaster in os.listdir(video_folder):
    broadcaster_path = os.path.join(video_folder, broadcaster)
    
    if not os.path.isdir(broadcaster_path):
        continue  # Skip if not a folder
    
    logger.info(f"Processing videos from: {broadcaster}")

    # Create broadcaster folder for audio if it doesn't exist
    broadcaster_audio_folder = os.path.join(audio_output_folder, broadcaster)
    os.makedirs(broadcaster_audio_folder, exist_ok=True)

    # Loop through each video file in the broadcaster's folder
    for video_file in tqdm(os.listdir(broadcaster_path)):
        if not video_file.lower().endswith((".mp4", ".mov")):  # Process mp4 and mov files
            continue

        video_path = os.path.join(broadcaster_path, video_file)

        try:
            # Load video using moviepy
            video_clip = VideoFileClip(video_path)

            # Extract audio from video
            audio_clip = video_clip.audio

            # Define audio file output path (save as .wav for Whisper)
            audio_file_name = f"{os.path.splitext(video_file)[0]}.wav"  # Save as .wav
            audio_file_path = os.path.join(broadcaster_audio_folder, audio_file_name)

            # Write the audio to a file (using .wav format)
            audio_clip.write_audiofile(audio_file_path)

            # Close the video and audio clips to free memory
            video_clip.close()
            audio_clip.close()

            logger.info(f"Audio extracted for {video_file} and saved to {audio_file_path}")

            # Step 2: Transcribe the audio using Whisper
            logger.info(f"Transcribing audio from {audio_file_name}")
            result = model.transcribe(audio_file_path)

            # Extract the transcribed text
            transcribed_text = result["text"]
            logger.info(f"Transcription for {video_file}:")
            logger.info(transcribed_text)

            # Optionally, save transcription to a text file
            transcription_file_path = os.path.join(broadcaster_audio_folder, f"{os.path.splitext(video_file)[0]}.txt")
            with open(transcription_file_path, "w") as f:
                f.write(transcribed_text)

            logger.info(f"Transcription saved to {transcription_file_path}")

        except Exception as e:
            logger.error(f"Error processing {video_file}: {e}")
            continue

logger.info("Audio extraction and transcription complete!")
