#!/Users/psylviana/Downloads/VideoAnalyse/YOLO-Video-Analysis/yolov8_env/bin/python

import os
import pandas as pd
import sys
from ultralytics import YOLO
import cv2  # OpenCV for frame handling
import time

# Define the path to the videos
VIDEO_FOLDER = "/Users/psylviana/Downloads/VideoAnalyse/VIDEOS"
OUTPUT_FOLDER = "/Users/psylviana/Downloads/VideoAnalyse/YOLO-Video-Analysis/OUTPUT_FRAMES"
CSV_SAVE_PATH = "/Users/psylviana/Downloads/VideoAnalyse/YOLO-Video-Analysis/detection_results_2sec.csv"

# Initialize YOLO model (using YOLOv8)
model = YOLO("yolov8n.pt")  # You can change to yolov8s.pt or other models

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Automatically detect broadcaster folders
broadcaster_folders = [f for f in os.listdir(VIDEO_FOLDER) if os.path.isdir(os.path.join(VIDEO_FOLDER, f))]

# Create a list to store detection results
detection_results = []

# Suppress console output from YOLO
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')  # Redirect output to null

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self._original_stdout  # Restore output


# Process videos for each broadcaster
total_videos = sum(len(os.listdir(os.path.join(VIDEO_FOLDER, folder))) for folder in broadcaster_folders)
processed_videos = 0

for folder in broadcaster_folders:
    folder_path = os.path.join(VIDEO_FOLDER, folder)

    # Get all video files (case-insensitive check for formats)
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp4", ".mov", ".avi", ".MOV"))]

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)

        # Open the video file with OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second

        # Validate FPS before proceeding
        if fps == 0 or fps is None:
            print(f"Warning: Unable to determine FPS for {video_file} !")
            cap.release()
            continue

        print(f"Processing {video_file} ({folder}) - {fps:.2f} FPS")
        
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Break when video ends

            frame_count += 1

            # Process one frame per second
            if frame_count % int(fps) == 0:
                with SuppressOutput():  # Suppress YOLO logging
                    results = model(frame)

                # Process results
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        detections = result.boxes.xywh.cpu().numpy()  # Bounding boxes (x, y, width, height)
                        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
                        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

                        # Convert detections to DataFrame
                        df = pd.DataFrame(detections, columns=['x', 'y', 'width', 'height'])
                        df['confidence'] = confidences
                        df['class_id'] = class_ids
                        df['Broadcaster'] = folder
                        df['Video'] = video_file
                        df['Frame'] = frame_count

                        detection_results.append(df)

                        # Save the detected frame with annotations (Optional)
                        annotated_image = result.plot()  # Draw boxes on the frame
                        output_video_folder = os.path.join(OUTPUT_FOLDER, folder, video_file.split('.')[0])
                        os.makedirs(output_video_folder, exist_ok=True)
                        frame_output_path = os.path.join(output_video_folder, f"frame_{frame_count:04d}.jpg")
                        cv2.imwrite(frame_output_path, annotated_image)

        cap.release()  # Release video capture object
        processed_videos += 1
        elapsed_time = time.time() - start_time

        print(f"Finished {video_file} ({processed_videos}/{total_videos}) in {elapsed_time:.2f} sec\n")

# Save detections to CSV
if detection_results:
    df = pd.concat(detection_results, ignore_index=True)

    if not df.empty:
        try:
            df.to_csv(CSV_SAVE_PATH, index=False)
            print(f"Results saved to {CSV_SAVE_PATH} with {len(df)} detections.")
        except Exception as e:
            print(f"Error saving CSV: {e}")
    else:
        print("No valid detections to save.")
else:
    print("No detections were made for any videos.")
