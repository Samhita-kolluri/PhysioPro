# ------------------------------------------------------
# Spacial and Timely alignment
# ------------------------------------------------------

import numpy as np
import snowflake.connector
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.linalg import orthogonal_procrustes
from scipy.interpolate import interp1d
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Snowflake configuration
SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USERNAME"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
}

# Connect to Snowflake
def connect_snowflake():
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)

# Fetch keypoints data
def fetch_keypoints_from_snowflake(video_name, table_name):
    conn = connect_snowflake()
    cursor = conn.cursor()
    
    query = f"""
    SELECT frame_number, keypoint_name, x, y, z
    FROM {table_name}
    WHERE video_name = '{video_name}'
    ORDER BY frame_number, keypoint_name;
    """
    
    cursor.execute(query)
    data = cursor.fetchall()
    
    keypoints_data = {}
    for row in data:
        frame_number, keypoint_name, x, y, z = row
        if frame_number not in keypoints_data:
            keypoints_data[frame_number] = {}
        keypoints_data[frame_number][keypoint_name] = np.array([x, y, z])
    
    cursor.close()
    conn.close()
    
    return keypoints_data

# Local coordinate system transformation (spatial normalization)
def compute_local_coordinate_system(keypoints):
    """ Compute the local coordinate system (using the hip joint center as the origin) """
    left_hip = keypoints["LEFT_HIP"]
    right_hip = keypoints["RIGHT_HIP"]
    left_shoulder = keypoints["LEFT_SHOULDER"]
    right_shoulder = keypoints["RIGHT_SHOULDER"]
    
    origin = (left_hip + right_hip) / 2
    x_axis = (right_hip - left_hip) / np.linalg.norm(right_hip - left_hip)
    y_axis = (right_shoulder - origin) / np.linalg.norm(right_shoulder - origin)
    z_axis = np.cross(x_axis, y_axis) # Ensure orthogonality
    z_axis /= np.linalg.norm(z_axis)  # Ensure it's a unit vector

    return origin, np.array([x_axis, y_axis, z_axis])

# Transform to local coordinate system
def transform_to_local_coordinates(keypoints, origin, axes):
    transformed = {}
    for kp_name, coords in keypoints.items():
        relative_pos = coords - origin
        transformed[kp_name] = np.dot(axes.T, relative_pos)  # Ensure matrix is correctly applied
    return transformed

# Rigid alignment (rotation alignment, no scaling)
def rigid_align(source, target):
    """ Use orthogonal Procrustes to compute the best rotation matrix (no scaling) """
    common_keys = set(source.keys()).intersection(set(target.keys()))
    source_matrix = np.array([source[k] for k in common_keys])
    target_matrix = np.array([target[k] for k in common_keys])

    R, _ = orthogonal_procrustes(source_matrix, target_matrix)
    aligned_source = {k: np.dot(R, v) for k, v in source.items()}
    return aligned_source

# Perform Dynamic Time Warping (DTW) for Temporal Alignment
def dtw_align(source_seq, target_seq):
    """ Apply DTW to align sequences of frames """
    
    # Flatten into 1D vectors (all keypoints per frame concatenated)
    source_seq = source_seq.reshape(source_seq.shape[0], -1)  # (num_frames, feature_dim)
    target_seq = target_seq.reshape(target_seq.shape[0], -1)  # (num_frames, feature_dim)
    
    # Compute Dynamic Time Warping (DTW)
    distance, path = fastdtw(source_seq, target_seq, dist=euclidean)

    # Align patient frames and fill missing frames
    aligned_source = {}
    filled_indices = set()

    for src_idx, tgt_idx in path:
        if src_idx < len(source_seq) and tgt_idx < len(target_seq):
            aligned_source[tgt_idx] = source_seq[src_idx]
            filled_indices.add(tgt_idx)

    # Fill missing frames (linear interpolation)
    for i in range(1, len(target_seq)):
        if i not in filled_indices:
            aligned_source[i] = aligned_source[i - 1]  # Use previous frame to fill
    
    # Reshape back to original shape (num_frames, num_keypoints, 3)
    aligned_source_array = np.array([aligned_source[i] for i in range(len(target_seq))])
    aligned_source_array = aligned_source_array.reshape(len(target_seq), -1, 3)

    return aligned_source_array

# Build data matrix
def build_vector_series(keypoints_dict):
    return np.array([np.vstack(list(frame.values())) for frame in keypoints_dict.values()])

# Main function
def process_videos(correct_video_name, patient_video_name, correct_table_name, patient_table_name):
    # Read data
    correct_video_keypoints = fetch_keypoints_from_snowflake(correct_video_name, correct_table_name)
    patient_video_keypoints = fetch_keypoints_from_snowflake(patient_video_name, patient_table_name)
    
    # Perform coordinate transformation frame by frame
    correct_local = {}
    patient_local = {}
    
    for frame in correct_video_keypoints:
        origin, axes = compute_local_coordinate_system(correct_video_keypoints[frame])
        correct_local[frame] = transform_to_local_coordinates(correct_video_keypoints[frame], origin, axes)

    for frame in patient_video_keypoints:
        origin, axes = compute_local_coordinate_system(patient_video_keypoints[frame])
        patient_local[frame] = transform_to_local_coordinates(patient_video_keypoints[frame], origin, axes)

    # Rotation alignment
    aligned_patient = {}
    all_frames = set(correct_local.keys()).union(set(patient_local.keys()))
    aligned_patient = {frame: rigid_align(patient_local.get(frame, patient_local[min(patient_local.keys())]),
                                          correct_local.get(frame, correct_local[min(correct_local.keys())]))
                       for frame in all_frames}

    # Temporal alignment (DTW)
    correct_series = build_vector_series(correct_local)
    patient_series = build_vector_series(aligned_patient)
    final_aligned_patient = dtw_align(patient_series, correct_series)

    return final_aligned_patient

# Interpolate keypoints to match patient video frame numbers
def interpolate_keypoints(final_aligned_patient, target_frames):
    """
    Interpolate the keypoint data from final_aligned_patient to the patient video frame numbers
    :param final_aligned_patient: Aligned keypoint data (485, 33, 3)
    :param target_frames: Target video frame count (536)
    :return: Interpolated keypoint data (536, 33, 3)
    """
    num_frames, num_keypoints, num_dims = final_aligned_patient.shape
    
    # Create interpolation function for each keypoint dimension
    interpolated_keypoints = np.zeros((target_frames, num_keypoints, num_dims))
    
    for keypoint_idx in range(num_keypoints):
        for dim_idx in range(num_dims):
            # Interpolate for each keypoint's dimension
            keypoints_dim = final_aligned_patient[:, keypoint_idx, dim_idx]
            
            # Create interpolation function, using linear interpolation
            interpolation_function = interp1d(np.arange(num_frames), keypoints_dim, kind='linear', fill_value="extrapolate")
            
            # Calculate new frame numbers (536 frames)
            interpolated_keypoints[:, keypoint_idx, dim_idx] = interpolation_function(np.linspace(0, num_frames - 1, target_frames))
    
    return interpolated_keypoints

# Perform alignment
final_aligned_patient = process_videos('lower_extremity/Fire_Hydrant', 'lower_extremity/patient_video', 'pose_keypoints', 'patient_pose_keypoints')

# Interpolate to match patient video frame numbers
final_interpolated_patient = interpolate_keypoints(final_aligned_patient, target_frames=536)

# Save the final data for overlay visualization
np.save("interpolated_patient_keypoints.npy", final_interpolated_patient)  # Ready for visualization

print("Data processing complete. Interpolation complete. The new shape of the keypoints:")
print("Shape:", final_interpolated_patient.shape)
