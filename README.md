# PhysioPro
Enhancing Mobility with Generative AI-Driven Motion Correction

PhysioPro is a physiotherapy assistance tool that aligns patient movement data with standard physiotherapy exercises, generates corrected movement visualizations using AI, and overlays them onto patient videos for feedback and guidance.

## Overview

The project consists of three main scripts:
1. **Spatial and Temporal Alignment**: Fetches keypoint data from Snowflake, aligns patient movements to a reference video using Procrustes analysis and Dynamic Time Warping (DTW), and interpolates the data.
2. **Normalization and Visualization**: Converts aligned keypoints to skeleton images, uses Stable Diffusion with ControlNet to generate corrected movement images, and overlays them onto the patient video.
3. **Frame Extraction**: Extracts frames from input videos for preprocessing (optional).

## Prerequisites

- **Python**: 3.10 or later
- **GPU (Optional)**: NVIDIA GPU with CUDA support for faster image generation
- **Snowflake Account**: For accessing keypoint data
- **Video Files**: Patient videos in `.mp4` format
- **Hugging Face Account**: For downloading Stable Diffusion models (optional token required)

### 1. Clone the Repository
```bash
git clone <https://github.com/Samhita-kolluri/PhysioPro>
cd PhysioPro
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Configure Environment Variables
```bash
SNOWFLAKE_USERNAME=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```
#### 4. Prepare Input Files

Place your patient video ( `patient_video.mp4`) in the project root.

Ensure videos for frame extraction are in `lower_extremity/videos/.`

### Outputs
interpolated_patient_keypoints.npy: Aligned and interpolated keypoint data (shape: 536, 33, 3).
output.mp4: Patient video with overlaid corrected movements.
lower_extremity/frames/: Extracted video frames (if diffusion_model.py is run).