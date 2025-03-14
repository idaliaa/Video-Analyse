#!/Users/psylviana/Downloads/VideoAnalyse/YOLO-Video-Analysis/yolov8_env/bin/python
import os
import ffmpeg
from tqdm import tqdm

# Path to main video folder
video_folder = "/Users/psylviana/Downloads/VideoAnalyse/VIDEOS"

# Path to store the extracted subtitle files
subtitle_output_folder = "/Users/psylviana/Downloads/VideoAnalyse/EXTRACTED_SUBTITLES"

# Create output folder if it doesn't exist
os.makedirs(subtitle_output_folder, exist_ok=True)

# Loop through each video file in the folder
for video_file in tqdm(os.listdir(video_folder)):
    if not video_file.endswith((".mp4", ".mov", ".avi", ".MOV", ".mpg")):  # Process video files
        continue

    video_path = os.path.join(video_folder, video_file)

    # Check if the video contains any subtitle streams using FFmpeg
    print(f"Checking subtitles for {video_file}...")

    try:
        # Run FFmpeg to get information about the video file
        probe = ffmpeg.probe(video_path, v='error', select_streams='s', show_entries='stream=codec_type:stream_tags')

        # Check if subtitle streams exist
        subtitle_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'subtitle']

        if not subtitle_streams:
            print(f"No subtitle streams found in {video_file}. Skipping.")
            continue  # Skip to next video if no subtitle streams are found

        # Loop through subtitle streams and extract each one
        for i, subtitle in enumerate(subtitle_streams):
            subtitle_file_name = f"{os.path.splitext(video_file)[0]}_subtitle_{i + 1}.srt"  # Save as .srt
            subtitle_file_path = os.path.join(subtitle_output_folder, subtitle_file_name)

            # Extract subtitle using FFmpeg
            print(f"Extracting subtitles from {video_file} to {subtitle_file_path}")
            try:
                # Check if subtitle stream is embedded
                ffmpeg.input(video_path, map=f"0:s:{subtitle['index']}").output(subtitle_file_path).run()
                print(f"Subtitle extracted and saved to {subtitle_file_path}")
            except ffmpeg.Error as e:
                print(f"Error extracting subtitles from {video_file}: {e}")

    except ffmpeg.Error as e:
        print(f"Error processing {video_file}: {e}")

print("Subtitle extraction complete!")
