#!/Users/psylviana/Downloads/VideoAnalyse/YOLO-Video-Analysis/yolov8_env/bin/python

import cv2
import insightface
import pandas as pd
import os
from ultralytics import YOLO

# Initialize YOLO model
yolo_model = YOLO("yolov8n.pt")  # Load the YOLO model

# Initialize the InsightFace model
face_model = insightface.app.FaceAnalysis()
face_model.prepare(ctx_id=0)  # 0 for CPU, 1 for GPU (if using GPU)

# Define the path to the image folder
image_folder = "/Users/psylviana/Downloads/VideoAnalyse/YOLO-Video-Analysis/OUTPUT_FRAMES"  # Update this path

# List image files (JPG files) in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg"))]

# Create an empty list to store the results
results = []

# Loop through each image file
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    print(f"Processing image: {image_file}")
    
    # Read the image using OpenCV
    frame = cv2.imread(image_path)
    
    # Check if the image is read correctly
    if frame is None:
        print(f"Error: Could not read image {image_file}. Skipping.")
        continue

    # Perform face detection using YOLO
    detections = yolo_model(frame)[0]  # Get YOLO detections
    faces = []

    for detection in detections.boxes:
        # Check if the detected class is 'face' (assuming YOLO detects faces as a class)
        if detection.cls == 0:  # '0' might be the class index for face (check YOLO class mapping)
            x1, y1, x2, y2 = map(int, detection.xywh[0])  # Get bounding box coordinates
            face_region = frame[y1:y2, x1:x2]  # Crop face region from the image
            faces.append(face_region)

    # Process each detected face with InsightFace
    for face in faces:
        # Get the faces and emotion details using InsightFace
        face_details = face_model.get(face)
        
        for face_detail in face_details:
            # Extract information about the face
            bbox = face_detail.bbox  # Bounding box (x1, y1, x2, y2)
            landmarks = face_detail.landmark  # Facial landmarks (if needed)
            emotion = face_detail.emotion  # Emotion detected by InsightFace

            # Prepare data for CSV
            for e in emotion:
                emotion_name = e['emotion']
                emotion_score = e['score']
                row = {
                    'Image': image_file,
                    'x1': int(bbox[0]),
                    'y1': int(bbox[1]),
                    'x2': int(bbox[2]),
                    'y2': int(bbox[3]),
                    'Emotion': emotion_name,
                    'Emotion_Score': emotion_score
                }
                results.append(row)

# Convert the results into a Pandas DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
output_csv = "/Users/psylviana/Downloads/VideoAnalyse/YOLO-Video-Analysis/emotion_results.csv"  # Specify where you want to save the CSV
df.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")
