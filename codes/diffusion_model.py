import numpy as np
import cv2
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import torchvision.transforms as transforms
import time

print(torch.cuda.is_available())  # Check if the GPU is enabled
print(torch.cuda.get_device_name(0))  # Display the GPU name

# === 1. Get video resolution ===
def get_video_size(video_path):
    """
    Read the width and height of the video
    :param video_path: Path to the patient video
    :return: Width and height of the video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None, None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return width, height

# === 2. Convert keypoints data to skeleton image ===
def keypoints_to_skeleton_image(keypoints_array, frame_size=(512, 512)):
    """
    Convert aligned keypoint data to a skeleton image
    :param keypoints_array: (536, 33, 3) Keypoint matrix
    :param frame_size: Image size
    :return: List of generated skeleton images
    """
    num_frames = keypoints_array.shape[0]
    skeleton_images = []
    
    for i in range(num_frames):
        frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
        keypoints = keypoints_array[i][:, :2]  # Take x, y directly
        
        # Get fixed minimum/maximum values to ensure the skeleton does not deform
        min_x, min_y = keypoints.min(axis=0)
        max_x, max_y = keypoints.max(axis=0)

        keypoints[:, 0] = (keypoints[:, 0] - min_x) / (max_x - min_x) * frame_size[0]
        keypoints[:, 1] = (keypoints[:, 1] - min_y) / (max_y - min_y) * frame_size[1]
        
        # Draw the keypoints
        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), -1)

        skeleton_images.append(Image.fromarray(frame))
    
    return skeleton_images

# === 3. Use ControlNet + Stable Diffusion to generate corrected images ===
def generate_corrected_images(skeleton_images):
    """
    Use Stable Diffusion + ControlNet to generate corrected images with standard movements
    :param skeleton_images: List of skeleton images
    :return: List of generated corrected movement images
    """
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    ).to("cuda")

    # **Disable NSFW filter**
    pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))

    corrected_images = []
    for skeleton in skeleton_images:
        image = pipe(
            prompt="A person in sportswear performing a correct physiotherapy movement, professional posture, high-quality, safe",
            negative_prompt="nude, nsfw, naked, inappropriate, revealing clothes",
            image=skeleton,
            num_inference_steps=15,
        ).images[0]
        corrected_images.append(image)

    return corrected_images

# === 4. Overlay corrected images on patient video ===
def overlay_corrected_images_on_video(video_path, corrected_images, output_path="output.mp4"):
    """
    Overlay the generated corrected images onto the patient video
    :param video_path: Path to the patient video
    :param corrected_images: List of corrected images
    :param output_path: Output video path
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(corrected_images):
            break

        # Resize corrected image to match video frame size
        corrected_img = corrected_images[frame_idx].resize((width, height))
        corrected_img_np = np.array(corrected_img)

        # Transparent blending (50% overlay)
        overlayed_frame = cv2.addWeighted(frame, 0.5, corrected_img_np, 0.5, 0)

        out.write(overlayed_frame)
        frame_idx += 1

    cap.release()
    out.release()

# === Main execution flow ===
if __name__ == "__main__":
    start_time = time.time()
    
    # Read the video resolution of the patient
    video_path = "patient_video.mp4"
    width, height = get_video_size(video_path)
    print(f"Patient video resolution: {width}x{height}")
    
    # Read the aligned keypoint data (shape: (536, 33, 3))
    aligned_keypoints = np.load("interpolated_patient_keypoints.npy")  

    # 1. Convert keypoints to skeleton images, using patient video resolution
    skeleton_images = keypoints_to_skeleton_image(aligned_keypoints, frame_size=(width, height))
    print(len(skeleton_images))
    
    # 2. Use ControlNet to generate corrected movement images
    corrected_images = generate_corrected_images(skeleton_images)
    print(len(corrected_images))

    # 3. Overlay onto the patient video
    overlay_corrected_images_on_video(video_path, corrected_images)

    end_time = time.time()
    print(f"Total generation time: {end_time - start_time:.2f} s")
