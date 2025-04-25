import streamlit as st
import snowflake.connector
import pandas as pd
from dotenv import load_dotenv
import os
import re
import time
from datetime import datetime
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL

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

# Create SQLAlchemy engine for Snowflake
try:
    engine = create_engine(URL(
        user=SNOWFLAKE_CONFIG["user"],
        password=SNOWFLAKE_CONFIG["password"],
        account=SNOWFLAKE_CONFIG["account"],
        warehouse=SNOWFLAKE_CONFIG["warehouse"],
        database=SNOWFLAKE_CONFIG["database"],
        schema=SNOWFLAKE_CONFIG["schema"],
    ))
except Exception as e:
    st.error(f"Failed to connect to Snowflake: {e}")
    st.stop()

# Connect to Snowflake for raw cursor operations
def connect_snowflake():
    try:
        return snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {e}")
        st.stop()

# Create uploads directory
UPLOAD_DIR = "./test_videos"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Streamlit UI setup
st.set_page_config(page_title="PhysioPro - Motion Feedback", layout="centered")
st.title("PhysioPro: AI Motion Correction & Feedback")

st.markdown("""
Welcome to **PhysioPro**, your virtual motion correction assistant.
Enter your personal details, upload your motion video, and receive AI-generated feedback based on motion analysis.
""")

# Step 1: Collect user personal details
st.markdown("### 🧑‍⚕️ Enter Your Personal Information")

body_area = st.selectbox(
    "Select Target Area of Exercise:",
    options=["Lower Body (Legs, Hips, Knees)", "Upper Body (Shoulders, Arms)", "Core (Abdomen, Back)"]
)
# Internally map to standardized type for storage
area_map = {
    "Lower Body (Legs, Hips, Knees)": "leg",
    "Upper Body (Shoulders, Arms)": "shoulders",
    "Core (Abdomen, Back)": "core"
}
exercise_type = area_map[body_area]
exercise_type = re.sub(r'\W+', '_', exercise_type.upper()) 
age = st.number_input("Age (years):", min_value=10, max_value=120, value=30)
weight = st.number_input("Weight (kg):", min_value=20, max_value=200, value=70)
height = st.number_input("Height (cm):", min_value=50, max_value=250, value=170)

# Display user details
# st.write(f"**Your Details:** Age: {age} years, Weight: {weight} kg, Height: {height} cm")
st.write(f"**Your Details:** Age: {age} years, Weight: {weight} kg, Height: {height} cm, Exercise: {exercise_type}")


# Step 2: Upload video
st.markdown("### 📤 Upload Your Motion Video")
video_file = st.file_uploader("Upload video (MP4/MOV)", type=["mp4", "mov"])

# Step 3 & 4: Show RMSE results, checkboxes, and LLM feedback after video upload
if video_file:
    # Save video locally with timestamp
    try:
        # Generate timestamp-based filename
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"patient_video.mp4"
        video_path = os.path.join(UPLOAD_DIR, video_filename)
        
        # Save video
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        st.success(f"Video uploaded and saved locally at {video_path}!")
    except Exception as e:
        st.error(f"Failed to save video locally: {e}")

    st.video(video_file)

   # Fetch RMSE data
    try:
        query = f"""
        SELECT keypoint_name, RMSE 
        FROM PHYSIOPRO_DB.PHYSIOPRO_SCHEMA.RMSE_RESULTS_{exercise_type.upper()}
    """
#        "SELECT keypoint_name, rmse FROM PHYSIOPRO_DB.PHYSIOPRO_SCHEMA.RMSE_RESULTS_{exercise_type}"
        df = pd.read_sql(query, engine)
        if df.empty:
        #   st.warning("No RMSE data found in RMSE_RESULTS table.")
           st.warning(f"No RMSE data found in RMSE_RESULTS_{exercise_type.upper()} table.")
    except Exception as e:
        st.error(f"Error fetching RMSE data: {e}")
        df = pd.DataFrame()

    # RMSE Results with Checkbox
    st.markdown("### 📊 Motion Analysis Results (RMSE)")
    show_rmse = st.checkbox("RMSE Values for Keypoints fetched from Snowflake:")
    if show_rmse and not df.empty:
        st.write("**RMSE Values for Keypoints:**")
        st.dataframe(df.rename(columns={"keypoint_name": "Keypoint", "rmse": "RMSE"}))
    elif show_rmse and df.empty:
        st.warning("No RMSE data to display.")

    # LLM Feedback with Prompt Checkbox
    st.markdown("### 🧠 AI Feedback")
    if not df.empty:
        # Calculate average RMSE
        avg_rmse = df['rmse'].mean()

        # Generate prompt for LLM
        # llm_prompt = f"""
        # You are a physiotherapy assistant. Based on the following data, provide feedback on the patient's motion stability and suggest exercises to improve their performance:
        # - Age: {age} years
        # - Weight: {weight} kg
        # - Height: {height} cm
        # - Average RMSE: {avg_rmse:.3f}
        # - Keypoint-specific RMSE values:
        # {', '.join([f"{row['keypoint_name']}: {row['rmse']:.3f}" for _, row in df.iterrows()])}
        # """

        # Generate prompt for LLM
        llm_prompt = f"""
        You are a physiotherapy assistant. Based on the following data, provide feedback on the patient's motion stability 
        and suggest physiotherapy-specific exercises to improve their performance in the selected focus area.
        Patient Information:
        - Age: {age} years
        - Weight: {weight} kg
        - Height: {height} cm
        - Target Area of Exercise: {exercise_type}
        - Average RMSE: {avg_rmse:.3f}
        - Keypoint-specific RMSE values:
        {', '.join([f"{row['keypoint_name']}: {row['rmse']:.3f}" for _, row in df.iterrows()])}
        """

        # Checkbox to show LLM prompt
        show_prompt = st.checkbox("Show LLM Prompt")
        if show_prompt:
            st.write("**Generated LLM Prompt:**")
            st.code(llm_prompt)

        # Delay LLM feedback by 5 seconds
        if 'video_upload_time' not in st.session_state:
            st.session_state.video_upload_time = time.time()

        elapsed_time = time.time() - st.session_state.video_upload_time
        if elapsed_time >= 5:
            # Query Cortex for feedback
            query = f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                'mistral_7b_PhysioPro',
                '{llm_prompt.replace("'", "''")}'
            ) AS feedback
            """
            try:
                conn = connect_snowflake()
                cur = conn.cursor()
                cur.execute(query)
                feedback = cur.fetchone()[0]
                st.success(feedback)
            except Exception as e:
                st.error(f"Error generating feedback: {e}")
            finally:
                conn.close()
        else:
            st.info(f"Generating AI feedback in {5 - int(elapsed_time)} seconds...")
    else:
        st.warning("No feedback available due to missing RMSE data.")
else:
    st.info("Please upload a video to view motion analysis and feedback.")

# Reset video upload time on new upload
if video_file and ('last_video' not in st.session_state or st.session_state.last_video != video_file.name):
    st.session_state.video_upload_time = time.time()
    st.session_state.last_video = video_file.name