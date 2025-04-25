import cv2
import numpy as np
import os
from PIL import Image
import time

# Define input and output directories
input_dir = "./lower_extremity/videos"
output_base_dir = "./lower_extremity/frames"
os.makedirs(output_base_dir, exist_ok=True)

# Get list of all video files
video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

start_time = time.time()

for video_file in video_files:
    video_path = os.path.join(input_dir, video_file)
    video_name = os.path.splitext(video_file)[0]  # Remove .mp4 extension

    # Create output directory for this video's frames
    output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        # Resize the frame
        resized_frame = cv2.resize(frame, (256, 256))

        # Normalize: Convert pixel values to range [0,1]
        normalized_frame = resized_frame.astype(np.float32) / 255.0

        # Convert BGR to RGB (OpenCV loads images in BGR format)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL image (if you need format conversion)
        pil_image = Image.fromarray(rgb_frame)

        # Save frame in corresponding subdirectory
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        pil_image.save(frame_filename)

        frame_count += 1

    # Release video capture
    cap.release()
    
    print(f"Extracted {frame_count} frames from '{video_file}' and saved in '{output_dir}'")

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Total processing time: {elapsed_time:.2f} s")