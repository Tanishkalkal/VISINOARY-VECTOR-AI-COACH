import os
import uuid
import subprocess
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

# import your push-up counter function
from pushup_counter import pushup_counter
from vertical_jump  import detect_jumps_autoheight
from sit_ups        import situp_counter
from sit_and_reach  import sit_and_reach_tracker

app = Flask(__name__)

# folders
UPLOAD_FOLDER = os.path.join("static", "uploads")
OUTPUT_FOLDER = os.path.join("static", "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# change file extentions as per requirement

ALLOWED_EXTS = {"mp4", "mov", "avi", "webm", "mkv"}

# ⚡ Full path to ffmpeg.exe (update if needed)
FFMPEG_PATH = "ffmpeg"

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def convert_webm_to_mp4(input_path, output_path):
    """
    Convert webm (or other input) to mp4 using ffmpeg if available.
    """
    cmd = [
        FFMPEG_PATH, "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

# fitness level functions or parameters for each test

def get_pushup_level(count, age, gender):
    
    """
    Determines the pushup fitness level based on age, gender, and count.
    """
    # Using the updated table for age ranges 14-17 and 20+
    if gender.lower() == 'male':
        if age >= 14 and age <= 17:
            if count >= 35:
                return "Excellent"
            elif count >= 18 and count <= 25:
                return "Average"
            else:
                return "Below Average"
        elif age >= 18 and age <= 19:
            if count >= 39:
                return "Excellent"
            elif count >= 22 and count <= 28:
                return "Average"
            else:
                return "Below Average"
        elif age >= 20:
            if count >= 30:
                return "Excellent"
            elif count >= 17 and count <= 21:
                return "Average"
            else:
                return "Below Average"
    elif gender.lower() == 'female':
        if age >= 14 and age <= 17:
            if count >= 28:
                return "Excellent"
            elif count >= 12 and count <= 18:
                return "Average"
            else:
                return "Below Average"
        elif age >= 18 and age <= 19:
            if count >= 33:
                return "Excellent"
            elif count >= 15 and count <= 20:
                return "Average"
            else:
                return "Below Average"
        elif age >= 20:
            if count >= 24:
                return "Excellent"
            elif count >= 12 and count <= 17:
                return "Average"
            else:
                return "Below Average"
    return "N/A" 

def get_jump_level(jump_height_cm, age, gender):
    """
    Determines the vertical jump fitness level based on age, gender, and jump height.
    """
    # Norms based on the provided image and common fitness standards
    jump_height_cm = float(jump_height_cm)
    if gender.lower() == 'male':
        if age >= 16 and age <= 19:
            if jump_height_cm >= 65:
                return "Excellent"
            elif jump_height_cm >= 50:
                return "Above Average"
            elif jump_height_cm >= 40:
                return "Average"
            elif jump_height_cm >= 30:
                return "Below Average"
            else:
                return "Poor"
        elif age >= 20:
            if jump_height_cm >= 70:
                return "Excellent"
            elif jump_height_cm >= 56:
                return "Above Average"
            elif jump_height_cm >= 41:
                return "Average"
            elif jump_height_cm >= 31:
                return "Below Average"
            else:
                return "Poor"
    elif gender.lower() == 'female':
        if age >= 16 and age <= 19:
            if jump_height_cm >= 58:
                return "Excellent"
            elif jump_height_cm >= 47:
                return "Above Average"
            elif jump_height_cm >= 36:
                return "Average"
            elif jump_height_cm >= 26:
                return "Below Average"
            else:
                return "Poor"
        elif age >= 20:
            if jump_height_cm >= 60:
                return "Excellent"
            elif jump_height_cm >= 46:
                return "Above Average"
            elif jump_height_cm >= 31:
                return "Average"
            elif jump_height_cm >= 21:
                return "Below Average"
            else:
                return "Poor"
    return "N/A"

def get_reach_level(reach_cm, age, gender):
    """
    Determines the sit-and-reach flexibility level based on age, gender, and reach in cm.
    """
    # Norms based on common fitness standards for younger adults
    reach_cm = float(reach_cm)
    if gender.lower() == 'male':
        if age >= 16 and age <= 19:
            if reach_cm >= 25: return "Excellent"
            elif reach_cm >= 20: return "Above Average"
            elif reach_cm >= 15: return "Average"
            else: return "Below Average"
        elif age >= 20 and age <= 30:
            if reach_cm >= 20: return "Excellent"
            elif reach_cm >= 15: return "Above Average"
            elif reach_cm >= 10: return "Average"
            else: return "Below Average"
    elif gender.lower() == 'female':
        if age >= 16 and age <= 19:
            if reach_cm >= 30: return "Excellent"
            elif reach_cm >= 25: return "Above Average"
            elif reach_cm >= 20: return "Average"
            else: return "Below Average"
        elif age >= 20 and age <= 30:
            if reach_cm >= 25: return "Excellent"
            elif reach_cm >= 20: return "Above Average"
            elif reach_cm >= 15: return "Average"
            else: return "Below Average"
    return "N/A"

def get_situp_level(count, age, gender):
    """
    Determines the sit-up fitness level based on age, gender, and count.
    Source: Norms from the President's Council on Physical Fitness and Sports
    """
    if gender.lower() == 'male':
        if age >= 16 and age <= 19:
            if count >= 45: return "Excellent"
            elif count >= 36: return "Above Average"
            elif count >= 29: return "Average"
            else: return "Below Average"
        elif age >= 20 and age <= 29:
            if count >= 49: return "Excellent"
            elif count >= 40: return "Above Average"
            elif count >= 34: return "Average"
            else: return "Below Average"
        elif age >= 30 and age <= 39:
            if count >= 41: return "Excellent"
            elif count >= 33: return "Above Average"
            elif count >= 27: return "Average"
            else: return "Below Average"
        else: return "N/A"
    elif gender.lower() == 'female':
        if age >= 16 and age <= 19:
            if count >= 42: return "Excellent"
            elif count >= 32: return "Above Average"
            elif count >= 25: return "Average"
            else: return "Below Average"
        elif age >= 20 and age <= 29:
            if count >= 44: return "Excellent"
            elif count >= 36: return "Above Average"
            elif count >= 28: return "Average"
            else: return "Below Average"
        elif age >= 30 and age <= 39:
            if count >= 38: return "Excellent"
            elif count >= 30: return "Above Average"
            elif count >= 24: return "Average"
            else: return "Below Average"
        else: return "N/A"
    return "N/A"

@app.route("/")
def index():
    """Serves the main HTML page. Assumes you have an index.html in a 'templates' folder."""
    # If your index.html is in the same directory, Flask won't find it by default.
    # It's best practice to create a 'templates' folder and put index.html inside it.
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_video():
    """
    A single, unified endpoint to handle all video uploads.
    It determines which test to run based on the 'test_type' form field.
    """
    if "video" not in request.files:
        return jsonify(error="No video file part in the request"), 400

    file = request.files["video"]
    age = request.form.get("age", type=int)
    gender = request.form.get("gender")
    test_type = request.form.get("test_type")

    if file.filename == "": return jsonify(error="No video selected"), 400
    if not all([age, gender, test_type]): return jsonify(error="Missing required form data"), 400
    if not allowed_file(file.filename): return jsonify(error="Invalid file type"), 400

    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_FOLDER, f"upload_{unique_id}_{filename}")
    file.save(input_path)

    converted_path = os.path.join(UPLOAD_FOLDER, f"conv_{unique_id}.mp4")
    if not convert_webm_to_mp4(input_path, converted_path):
        return jsonify(error="Video conversion failed. Check FFMPEG path and file integrity."), 500

    output_path = os.path.join(OUTPUT_FOLDER, f"output_{unique_id}.mp4")
    results = {}
    
    try:
        # This block calls the correct function based on the test_type from the frontend
        if test_type == "pushups":
            count, final_path = pushup_counter(converted_path, output_path)
            results = {"score_type": "Push-ups", "score": count, "level": get_pushup_level(count, age, gender)}
        
        elif test_type == "situps":
            valid, bad, final_path = situp_counter(converted_path, output_path)
            results = {"score_type": "Sit-ups", "score": valid, "level": get_situp_level(valid, age, gender), "secondary_score": bad, "secondary_score_label": "Bad Reps"}

        elif test_type == "sit_and_reach":
            reach, final_path = sit_and_reach_tracker(converted_path, output_path)
            results = {"score_type": "Sit and Reach", "score": f"{reach:.1f} cm", "level": get_reach_level(reach, age, gender)}

        elif test_type == "vertical_jump":
            final_path, heights = detect_jumps_autoheight(converted_path, output_path)
            avg_jump = round(np.mean(heights), 1) if heights else 0
            results = {"score_type": "Vertical Jump", "score": f"{avg_jump} cm", "level": get_jump_level(avg_jump, age, gender)}
        
        else:
            return jsonify(error=f"Unknown test type: {test_type}"), 400
            
        # The '_external=True' is important for the frontend to get the full URL
        results['video_url'] = url_for("static", filename=f"outputs/{os.path.basename(final_path)}", _external=True)

    except Exception as e:
        print(f"Error processing {test_type}: {e}")
        return jsonify(error="An error occurred during video analysis."), 500

    # Return a single, consistent JSON object to the frontend
    return jsonify(results)


if __name__ == "__main__":
    app.run(port=8001, debug=True)
    
    
    
    
    
    
    
    