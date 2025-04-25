import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import snowflake.connector
from dotenv import load_dotenv
import os
import cv2
import time
import json
import re
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

load_dotenv()

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USERNAME"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
}

def connect_snowflake():
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)

def fetch_keypoints_from_snowflake(video_name, table_name):
    conn = connect_snowflake()
    cursor = conn.cursor()
    
    query = f"""
    SELECT frame_number, keypoint_name, x, y, z
    FROM {table_name}
    WHERE video_name = '{video_name}';
    """
    cursor.execute(query)
    data = cursor.fetchall()
    
    keypoints_data = {}
    for row in data:
        frame_number, keypoint_name, x, y, z = row
        if frame_number not in keypoints_data:
            keypoints_data[frame_number] = {}
        keypoints_data[frame_number][keypoint_name] = np.array([x, y, z], dtype=np.float32)
    
    cursor.close()
    conn.close()
    
    return keypoints_data

def extract_pose_keypoints(video_file, local_video_path):
    """Process video to extract pose keypoints using MediaPipe."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(local_video_path)
    keypoints_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            frame_keypoints = {
                landmark.name: {
                    "x": results.pose_landmarks.landmark[landmark].x,
                    "y": results.pose_landmarks.landmark[landmark].y,
                    "z": results.pose_landmarks.landmark[landmark].z
                } for landmark in mp_pose.PoseLandmark
            }
            keypoints_data.append(frame_keypoints)

    cap.release()
    return keypoints_data

def save_keypoints_to_json(keypoints, output_path):
    with open(output_path, 'w') as f:
        json.dump(keypoints, f)

def load_keypoints_from_json(input_path):
    with open(input_path, 'r') as f:
        patient_data_raw = json.load(f)
        patient_keypoint_data = {
            int(frame_idx): {
                kp_name: np.array([
                    float(kp_values["x"]),
                    float(kp_values["y"]),
                    float(kp_values["z"])
                ], dtype=np.float32)
                for kp_name, kp_values in keypoints.items()
            }
            for frame_idx, keypoints in enumerate(patient_data_raw)
        }
    return patient_keypoint_data

# ============ Skeleton Direction Alignment (Local Alignment to a Standard Coordinate System) ============
def align_skeleton_to_standard(keypoints_data):
    """
    Align the keypoints data to a "standard human coordinate system":
      - X-axis: from the midpoint of the hips (hip_mid) to the midpoint of the shoulders (shoulder_mid) => +X direction
      - Y-axis: from left shoulder to right shoulder => +Y
      - Forward: -Z-axis
    
    Returns a new dictionary with the same structure as the input, but with coordinates shifted and rotated.
    """
    # Custom standard coordinate system's three basis vectors
    standard_x = np.array([-1, 0, 0], dtype=np.float32)
    standard_y = np.array([0, 1, 0], dtype=np.float32)
    standard_z = np.array([0, 0, -1], dtype=np.float32)

    aligned_data = {}

    for frame_idx, frame_dict in keypoints_data.items():
        
        needed_keys = ["LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
        if not all(k in frame_dict for k in needed_keys):
            aligned_data[frame_idx] = frame_dict
            continue
        
        left_hip = frame_dict["LEFT_HIP"]
        right_hip = frame_dict["RIGHT_HIP"]
        hip_mid = (left_hip + right_hip) / 2.0

        left_shoulder = frame_dict["LEFT_SHOULDER"]
        right_shoulder = frame_dict["RIGHT_SHOULDER"]
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0

        # Local coordinates
        local_x = shoulder_mid - hip_mid  # Upward
        if np.linalg.norm(local_x) < 1e-6:
            aligned_data[frame_idx] = frame_dict
            continue
        local_x /= np.linalg.norm(local_x)

        local_y = right_shoulder - left_shoulder  # Left -> Right
        if np.linalg.norm(local_y) < 1e-6:
            aligned_data[frame_idx] = frame_dict
            continue
        local_y /= np.linalg.norm(local_y)

        local_z = np.cross(local_x, local_y)  # Backward
        if np.linalg.norm(local_z) < 1e-6:
            aligned_data[frame_idx] = frame_dict
            continue
        local_z /= np.linalg.norm(local_z)

        M_local = np.stack([local_x, local_y, local_z], axis=1)
        M_std = np.stack([standard_x, standard_y, standard_z], axis=1)
        R = M_std @ M_local.T

        new_frame_dict = {}
        for k_name, coord in frame_dict.items():
            shifted = coord - hip_mid
            rotated = R @ shifted
            new_frame_dict[k_name] = rotated
        
        aligned_data[frame_idx] = new_frame_dict

    return aligned_data

# ============ Compute Local Coordinate Matrix for One Frame for "Single Global Rotation" ============
def compute_local_axes_for_frame(frame_dict):
    needed_keys = ["LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
    if not all(k in frame_dict for k in needed_keys):
        return None
    
    left_hip = frame_dict["LEFT_HIP"]
    right_hip = frame_dict["RIGHT_HIP"]
    hip_mid = (left_hip + right_hip) / 2.0
    
    left_shoulder = frame_dict["LEFT_SHOULDER"]
    right_shoulder = frame_dict["RIGHT_SHOULDER"]
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0

    local_x = shoulder_mid - hip_mid
    if np.linalg.norm(local_x) < 1e-6:
        return None
    local_x /= np.linalg.norm(local_x)

    local_y = right_shoulder - left_shoulder
    if np.linalg.norm(local_y) < 1e-6:
        return None
    local_y /= np.linalg.norm(local_y)

    local_z = np.cross(local_x, local_y)
    if np.linalg.norm(local_z) < 1e-6:
        return None
    local_z /= np.linalg.norm(local_z)

    M_local = np.stack([local_x, local_y, local_z], axis=1)
    return M_local

def single_global_rotation(aligned_correct, aligned_patient):
    cframe = min(aligned_correct.keys())
    pframe = min(aligned_patient.keys())

    if cframe not in aligned_correct or pframe not in aligned_patient:
        return np.eye(3, dtype=np.float32)

    M_c = compute_local_axes_for_frame(aligned_correct[cframe])
    M_p = compute_local_axes_for_frame(aligned_patient[pframe])
    if M_c is None or M_p is None:
        return np.eye(3, dtype=np.float32)

    R = M_p @ M_c.T
    return R

def apply_global_rotation_to_dict(keypoints_data, R):
    rotated_dict = {}
    for fidx, frame_dict in keypoints_data.items():
        new_frame = {}
        for k_name, coord in frame_dict.items():
            new_frame[k_name] = R @ coord 
        rotated_dict[fidx] = new_frame
    return rotated_dict

# ============ Convert aligned skeleton to NumPy sequence ============
def dictionary_to_frame_array(keypoints_dict, keypoints_order=None):
    if keypoints_order is None:
        # Retrieve the keypoint order from mediapipe Pose
        mp_pose = mp.solutions.pose
        keypoints_order = [landmark.name for landmark in mp_pose.PoseLandmark]
    
    frames_sorted = sorted(keypoints_dict.keys())
    
    array_list = []
    for fidx in frames_sorted:
        kp_coords = []
        frame_data = keypoints_dict[fidx]
        for kname in keypoints_order:
            if kname in frame_data:
                kp_coords.append(frame_data[kname])
            else:
                kp_coords.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
        array_list.append(kp_coords)
    
    arr = np.array(array_list)  # shape=(num_frames, num_keypoints, 3)
    return arr, frames_sorted, keypoints_order

# ============ DTW Time Warping (Time Dimension) ============
def dtw_time_warping(source_array, target_array):
    sN, sK, s3 = source_array.shape
    tN, tK, t3 = target_array.shape
    source_flat = source_array.reshape(sN, sK * s3)
    target_flat = target_array.reshape(tN, tK * t3)

    _, path = fastdtw(source_flat, target_flat, dist=euclidean)
    
    aligned_source = np.zeros((tN, sK, s3), dtype=np.float32)
    
    fill_count = {}
    for (si, ti) in path:
        aligned_source[ti, :, :] = source_array[si, :, :]
        fill_count[ti] = fill_count.get(ti, 0) + 1

    for i in range(tN):
        if i not in fill_count:
            if i > 0 and i-1 in fill_count:
                aligned_source[i] = aligned_source[i-1]
                fill_count[i] = 1
            else:
                j = i+1
                while j < tN and j not in fill_count:
                    j += 1
                if j < tN:
                    aligned_source[i] = aligned_source[j]
                    fill_count[i] = 1
                else:
                    pass
    
    return aligned_source

# ============ Overlay Two Skeletons in the Same Animation ============
def draw_overlay_skeleton_animation(arr_correct, arr_patient, keypoints_order, save_path="overlay_skeleton_animation.gif"):
    
    # Define skeleton connections (limbs)
    limbs = [
        ('NOSE', 'LEFT_EYE_INNER'), ('LEFT_EYE_INNER', 'LEFT_EYE'), ('LEFT_EYE', 'LEFT_EYE_OUTER'), 
        ('NOSE', 'RIGHT_EYE_INNER'), ('RIGHT_EYE_INNER', 'RIGHT_EYE'), ('RIGHT_EYE', 'RIGHT_EYE_OUTER'),
        ('NOSE', 'MOUTH_LEFT'), ('NOSE', 'MOUTH_RIGHT'),
        
        ('LEFT_EAR', 'LEFT_SHOULDER'), ('RIGHT_EAR', 'RIGHT_SHOULDER'),
        
        ('LEFT_SHOULDER', 'LEFT_ELBOW'), ('LEFT_ELBOW', 'LEFT_WRIST'), 
        ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'),
        
        ('LEFT_SHOULDER', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'RIGHT_HIP'),
        
        ('LEFT_HIP', 'LEFT_KNEE'), ('LEFT_KNEE', 'LEFT_ANKLE'), 
        ('RIGHT_HIP', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_ANKLE'),
        
        ('LEFT_ANKLE', 'LEFT_HEEL'), ('RIGHT_ANKLE', 'RIGHT_HEEL'), 
        ('LEFT_HEEL', 'LEFT_FOOT_INDEX'), ('RIGHT_HEEL', 'RIGHT_FOOT_INDEX')
    ]
    kp_index_map = {kname: i for i, kname in enumerate(keypoints_order)}

    F = arr_correct.shape[0]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-60)

    # Calculate coordinate ranges
    x_vals_all = np.concatenate([arr_correct[...,0].ravel(), arr_patient[...,0].ravel()])
    y_vals_all = np.concatenate([arr_correct[...,1].ravel(), arr_patient[...,1].ravel()])
    z_vals_all = np.concatenate([arr_correct[...,2].ravel(), arr_patient[...,2].ravel()])

    x_min, x_max = np.nanmin(x_vals_all), np.nanmax(x_vals_all)
    y_min, y_max = np.nanmin(y_vals_all), np.nanmax(y_vals_all)
    z_min, z_max = np.nanmin(z_vals_all), np.nanmax(z_vals_all)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Scatter plots: correct=blue, patient=orange
    scat_correct = ax.scatter([], [], [], c='b', marker='o', s=20)
    scat_patient = ax.scatter([], [], [], c='orange', marker='o', s=20)

    # Skeleton connections (lines)
    lines_correct = [ax.plot([], [], [], 'b', lw=2)[0] for _ in range(len(limbs))]
    lines_patient = [ax.plot([], [], [], 'orange', lw=2)[0] for _ in range(len(limbs))]

    def update(frame_idx):
        ccoords = arr_correct[frame_idx]  # shape=(K,3)
        pcoords = arr_patient[frame_idx]  # shape=(K,3)

        scat_correct._offsets3d = (ccoords[:,0], ccoords[:,1], ccoords[:,2])
        scat_patient._offsets3d = (pcoords[:,0], pcoords[:,1], pcoords[:,2])

        # Correct skeleton connections (lines)
        for i, (start, end) in enumerate(limbs):
            if (start in kp_index_map) and (end in kp_index_map):
                si = kp_index_map[start]
                ei = kp_index_map[end]
                xs = [ccoords[si,0], ccoords[ei,0]]
                ys = [ccoords[si,1], ccoords[ei,1]]
                zs = [ccoords[si,2], ccoords[ei,2]]
                lines_correct[i].set_data(xs, ys)
                lines_correct[i].set_3d_properties(zs)

        # Patient skeleton connections (lines)
        for i, (start, end) in enumerate(limbs):
            if (start in kp_index_map) and (end in kp_index_map):
                si = kp_index_map[start]
                ei = kp_index_map[end]
                xs = [pcoords[si,0], pcoords[ei,0]]
                ys = [pcoords[si,1], pcoords[ei,1]]
                zs = [pcoords[si,2], pcoords[ei,2]]
                lines_patient[i].set_data(xs, ys)
                lines_patient[i].set_3d_properties(zs)

        return (scat_correct, scat_patient) + tuple(lines_correct) + tuple(lines_patient)

    ani = animation.FuncAnimation(fig, update, frames=F, interval=30, blit=False)
    ani.save(save_path, writer=PillowWriter(fps=30))
    print(f"Overlay animation saved as: {save_path}")

# ============ Compute Numeric Metrics ============
def compute_3d_coordinate_rmse_for_keypoints(correct_arr, patient_arr, keypoints):
    # Get the index of the keypoints
    name_to_idx = {name: i for i, name in enumerate(kp_order)}

    # Ensure all keypoints are in kp_order
    if not all(kp in name_to_idx for kp in keypoints):
        print("[Warning] compute_3d_coordinate_rmse_for_keypoints: some keypoints are missing.")
        return np.nan

    # Calculate RMSE for each keypoint
    keypoint_rmse_list = []
    for keypoint in keypoints:
        keypoint_idx = name_to_idx[keypoint]

        # Calculate the error for this keypoint
        diff = patient_arr[:, keypoint_idx] - correct_arr[:, keypoint_idx]
        diff_sq = diff ** 2
        mse = np.nanmean(diff_sq)  # MSE across all frames
        rmse = np.sqrt(mse)  # Get RMSE
        keypoint_rmse_list.append(rmse)

    # Return the RMSE for all keypoints
    return keypoint_rmse_list

def angle_between_vectors(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_val = np.dot(v1, v2) / (norm1 * norm2)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    angle = np.arccos(cos_val)
    return np.degrees(angle)

def compute_hip_abduction_angle(hip, knee):
    v1 = knee - hip  # Vector from hip to knee
    v2 = np.array([0, 1, 0])  # Assume Y-axis is vertically upwards
    return angle_between_vectors(v1, v2)

def compute_knee_angle(hip, knee, ankle):
    v1 = hip - knee
    v2 = ankle - knee
    return angle_between_vectors(v1, v2)

def compute_knee_angle_rmse(correct_arr, patient_arr, kp_order):
    name_to_idx = {name: i for i, name in enumerate(kp_order)}

    needed = ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", 
              "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
    if not all(n in name_to_idx for n in needed):
        print("[Warning] compute_knee_angle_rmse: missing some keypoints.")
        return np.nan

    lh_idx = name_to_idx["LEFT_HIP"]
    lk_idx = name_to_idx["LEFT_KNEE"]
    la_idx = name_to_idx["LEFT_ANKLE"]
    rh_idx = name_to_idx["RIGHT_HIP"]
    rk_idx = name_to_idx["RIGHT_KNEE"]
    ra_idx = name_to_idx["RIGHT_ANKLE"]

    # Calculate for each frame
    left_angle_diff_sq = []
    right_angle_diff_sq = []
    left_hip_abduction_diff_sq = []
    right_hip_abduction_diff_sq = []

    F = correct_arr.shape[0]
    for f in range(F):
        c_l_hip, c_l_knee, c_l_ankle = correct_arr[f, lh_idx], correct_arr[f, lk_idx], correct_arr[f, la_idx]
        p_l_hip, p_l_knee, p_l_ankle = patient_arr[f, lh_idx], patient_arr[f, lk_idx], patient_arr[f, la_idx]
        c_r_hip, c_r_knee, c_r_ankle = correct_arr[f, rh_idx], correct_arr[f, rk_idx], correct_arr[f, ra_idx]
        p_r_hip, p_r_knee, p_r_ankle = patient_arr[f, rh_idx], patient_arr[f, rk_idx], patient_arr[f, ra_idx]

        # Calculate knee angle
        c_l_angle = compute_knee_angle(c_l_hip, c_l_knee, c_l_ankle)
        p_l_angle = compute_knee_angle(p_l_hip, p_l_knee, p_l_ankle)
        c_r_angle = compute_knee_angle(c_r_hip, c_r_knee, c_r_ankle)
        p_r_angle = compute_knee_angle(p_r_hip, p_r_knee, p_r_ankle)

        # Calculate hip abduction angle
        c_l_hip_abduction = compute_hip_abduction_angle(c_l_hip, c_l_knee)
        p_l_hip_abduction = compute_hip_abduction_angle(p_l_hip, p_l_knee)
        c_r_hip_abduction = compute_hip_abduction_angle(c_r_hip, c_r_knee)
        p_r_hip_abduction = compute_hip_abduction_angle(p_r_hip, p_r_knee)

        # Accumulate squared error
        left_angle_diff_sq.append((c_l_angle - p_l_angle)**2)
        right_angle_diff_sq.append((c_r_angle - p_r_angle)**2)
        left_hip_abduction_diff_sq.append((c_l_hip_abduction - p_l_hip_abduction)**2)
        right_hip_abduction_diff_sq.append((c_r_hip_abduction - p_r_hip_abduction)**2)

    # Calculate MSE for each angle
    left_knee_mse = np.mean(left_angle_diff_sq)
    right_knee_mse = np.mean(right_angle_diff_sq)
    left_hip_abduction_mse = np.mean(left_hip_abduction_diff_sq)
    right_hip_abduction_mse = np.mean(right_hip_abduction_diff_sq)

    # Calculate RMSE
    left_knee_rmse = np.sqrt(left_knee_mse)
    right_knee_rmse = np.sqrt(right_knee_mse)
    left_hip_abduction_rmse = np.sqrt(left_hip_abduction_mse)
    right_hip_abduction_rmse = np.sqrt(right_hip_abduction_mse)

    # Return the average RMSE for left and right knees, and hip abduction angles
    total_knee_rmse = 0.5 * (left_knee_rmse + right_knee_rmse)
    total_hip_abduction_rmse = 0.5 * (left_hip_abduction_rmse + right_hip_abduction_rmse)

    return total_knee_rmse, total_hip_abduction_rmse

def upload_read_rmse_from_snowflake(keypoints_to_compute_rmse, 
                                  keypoint_rmse, 
                                  knee_angle_rmse, 
                                  hip_abduction_angle_rmse, 
                                  exercise_type):
    data = {
        "keypoint_name": keypoints_to_compute_rmse + ["KNEE_ANGLE", "HIP_ABDUCTION_ANGLE"],
        "RMSE": keypoint_rmse + [knee_angle_rmse, hip_abduction_angle_rmse]
    }
    df = pd.DataFrame(data)

    conn = connect_snowflake()
    cursor = conn.cursor()

    exercise_type = re.sub(r'\W+', '_', exercise_type.upper())  # Only alphanumeric and underscores

    create_table_query = f'''
    CREATE TABLE IF NOT EXISTS RMSE_RESULTS_{exercise_type} (
        keypoint_name STRING,
        RMSE FLOAT,
        TYPE STRING
    );
    '''
    cursor.execute(create_table_query)
    
    truncate_table_query = f'''
    TRUNCATE TABLE RMSE_RESULTS_{exercise_type};
    '''
    cursor.execute(truncate_table_query)
    
    insert_query = f"""
    INSERT INTO RMSE_RESULTS_{exercise_type} (keypoint_name, RMSE, TYPE) 
    VALUES (%s, %s, %s)
    """
    for _, row in df.iterrows():
        cursor.execute(insert_query, (row['keypoint_name'], row['RMSE'], exercise_type))

    conn.commit()
    print(f"RMSE results imported to Snowflake table RMSE_RESULTS_{exercise_type}")

    query = f"""
    SELECT *
    FROM RMSE_RESULTS_{exercise_type}
    where TYPE = '{exercise_type}';
    """
    cursor.execute(query)
    data = cursor.fetchall()

    cursor.close()
    conn.close()
    return data

# ============ Main ============
# if __name__ == "__main__":
def main(exercise_type):
    """
    Main function to process a video, compute keypoints, and generate metrics.
    
    Args:
        exercise_type (str): Type of exercise (e.g., 'LEG', 'SHOULDERS', 'CORE')
    
    Returns:
        dict: Results containing animation path and processing time
    """
    start_time = time.time()
    correct_video_keypoints = fetch_keypoints_from_snowflake(
        'lower_extremity/resized_correct_video', 
        'pose_keypoints_new')
    
    # Specify video path and keypoint JSON output path
    video_filename = 'patient_video.mp4'
    video_path = os.path.join('test_videos', video_filename)
    keypoints_json_path = os.path.join('test_keypoints', 'keypoints.json')

    # Step 1: Extract keypoints and save them as JSON
    keypoints = extract_pose_keypoints(video_filename, video_path)
    save_keypoints_to_json(keypoints, keypoints_json_path)

    # Load JSON as patient_video_keypoints
    patient_video_keypoints = load_keypoints_from_json(keypoints_json_path)
    
    aligned_correct = align_skeleton_to_standard(correct_video_keypoints)
    aligned_patient = align_skeleton_to_standard(patient_video_keypoints)

    R_global = single_global_rotation(aligned_correct, aligned_patient)
    aligned_correct = apply_global_rotation_to_dict(aligned_correct, R_global)

    correct_arr, correct_frames, kp_order = dictionary_to_frame_array(aligned_correct)
    patient_arr, patient_frames, _ = dictionary_to_frame_array(aligned_patient, kp_order)

    # Perform time-alignment using DTW
    patient_aligned_arr = dtw_time_warping(patient_arr, correct_arr)

    # Overlay visualization
    draw_overlay_skeleton_animation(arr_correct=correct_arr,
                                    arr_patient=patient_aligned_arr,
                                    keypoints_order=kp_order,
                                    save_path="test_skeleton_animation/normalized_overlay_skeleton_animation_0525.gif")

    # Specify the keypoints to compute RMSE for
    keypoints_to_compute_rmse = ['LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
                                 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    
    keypoint_rmse = compute_3d_coordinate_rmse_for_keypoints(correct_arr, patient_aligned_arr, keypoints_to_compute_rmse)
    knee_angle_rmse, hip_abduction_angle_rmse = compute_knee_angle_rmse(correct_arr, patient_aligned_arr, kp_order)

    exercise_type = 'LEG'
    data = upload_read_rmse_from_snowflake(keypoints_to_compute_rmse,
                                         keypoint_rmse, knee_angle_rmse, hip_abduction_angle_rmse, exercise_type)

    print(data)

    end_time = time.time()
    processing_time = end_time - start_time
    return {
        "processing_time": processing_time
    }
#    print(f"Total time: {'processing_time':.2f} seconds.")


if __name__ == "__main__":
    # For testing purposes
    exercise_type = 'LEG'
    results = main(exercise_type)
    print(f"Total time: {results['processing_time']:.2f} seconds.")