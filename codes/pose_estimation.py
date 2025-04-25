import cv2
import mediapipe as mp
import os
import json
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils  # For visualization

# Define input and output directories
input_dir = "./lower_extremity/videos"
output_dir = "./lower_extremity/keypoints"
os.makedirs(output_dir, exist_ok=True)

# Get list of all video files
video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

start_time = time.time()

for video_file in video_files:
    video_path = os.path.join(input_dir, video_file)
    video_name = os.path.splitext(video_file)[0]  # Remove .mp4 extension

    # JSON file to store keypoints
    keypoints_file = os.path.join(output_dir, f"{video_name}_keypoints.json")
    keypoints_data = []  # Store keypoints for all frames

    # Open video
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB (MediaPipe uses RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with Pose model
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Extract keypoints as a dictionary
            frame_keypoints = {
                landmark.name: {"x": results.pose_landmarks.landmark[landmark].x,
                                "y": results.pose_landmarks.landmark[landmark].y,
                                "z": results.pose_landmarks.landmark[landmark].z}
                for landmark in mp_pose.PoseLandmark
            }

            keypoints_data.append(frame_keypoints)

    cap.release()

    # Save keypoints to a JSON file
    with open(keypoints_file, "w") as f:
        json.dump(keypoints_data, f, indent=4)

    print(f"Extracted keypoints saved to {keypoints_file}")

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Total processing time: {elapsed_time:.2f} s")
