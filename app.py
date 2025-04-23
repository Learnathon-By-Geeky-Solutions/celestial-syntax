from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
import sqlite3
import base64
import cv2
import numpy as np
import json
import subprocess
import re # Make sure re is imported

facial_recognition_process = None
app = Flask(__name__)
# You MUST set a secret key for session management
app.secret_key = os.urandom(24) # Generates a random secret key

FACE_IMAGES_DIR = "data/data_faces_from_camera/"
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)

# --- Routes from app.pdf (ensure they are included and imports are correct) ---
@app.route('/')
def index():
    # Your existing index route logic from app.pdf [cite: 852]
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM courses") # [cite: 852]
    courses = cursor.fetchall() # [cite: 852]
    conn.close()
    return render_template('index.html', selected_date='', no_data=False, courses=courses) # [cite: 852]

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
     # Your existing attendance route logic from app.pdf [cite: 852, 853, 854]
     # Make sure flash, redirect, url_for are imported
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM courses") # [cite: 852]
    courses = cursor.fetchall() # [cite: 852]
    if request.method == 'POST':
        selected_date = request.form.get('selected_date') # [cite: 852]
        course_id = request.form.get('course_id') # [cite: 852]
        if not selected_date or not course_id: # [cite: 852]
            flash("Please select both a date and a course.") # [cite: 852]
            return redirect(url_for('attendance')) # [cite: 852]
        try:
            cursor.execute("""
                SELECT s.roll_number, s.name, c.name, a.present
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                JOIN courses c ON a.course_id = c.id
                WHERE a.date = ? AND a.course_id = ?
                """, (selected_date, course_id)) # [cite: 853, 854]
            attendance_data = cursor.fetchall() # [cite: 854]
            if not attendance_data: # [cite: 854]
                 return render_template('index.html', selected_date=selected_date, no_data=True, courses=courses) # [cite: 854]
            else:
                 return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data, courses=courses) # [cite: 854]
        except Exception as e:
             flash(f"An error occurred: {str(e)}") # [cite: 854]
             return redirect(url_for('attendance')) # [cite: 854]
        finally:
             conn.close() # [cite: 854]
    else:
         conn.close() # [cite: 854]
         return render_template('index.html', selected_date=None, no_data=False, courses=courses) # [cite: 854]


@app.route('/course_attendance', methods=['GET'])
def course_attendance():
     # Your existing course_attendance route logic from app.pdf [cite: 854, 855]
    course_id = request.args.get('course_id') # [cite: 854]
    start_date = request.args.get('start_date') # [cite: 854]
    end_date = request.args.get('end_date') # [cite: 854]
    if not course_id or not start_date or not end_date: # [cite: 854]
        return "Please select a course and a valid date range.", 400 # [cite: 854]
    conn = sqlite3.connect('attendance.db') # [cite: 854]
    cursor = conn.cursor() # [cite: 854]
    cursor.execute('''
        SELECT s.roll_number, s.name, COUNT(a.id) AS total_classes, SUM(a.present) AS total_present
        FROM students s
        LEFT JOIN attendance a ON s.id = a.student_id AND a.course_id = ? AND a.date BETWEEN ? AND ?
        GROUP BY s.id
        ''', (course_id, start_date, end_date)) # [cite: 855]
    students_attendance = cursor.fetchall() # [cite: 855]
    students_data = [] # [cite: 855]
    for student in students_attendance: # [cite: 855]
        roll_number, name, total_classes, total_present = student # [cite: 855]
        total_classes = total_classes or 0 # [cite: 855]
        total_present = total_present or 0 # [cite: 855]
        attendance_percentage = (total_present / total_classes) * 100 if total_classes > 0 else 0 # [cite: 855]
        can_sit_exam = attendance_percentage >= 50 # [cite: 855]
        students_data.append({ # [cite: 855]
            'roll_number': roll_number, # [cite: 855]
            'name': name, # [cite: 855]
            'total_classes': total_classes, # [cite: 855]
            'total_present': total_present, # [cite: 855]
            'attendance_percentage': round(attendance_percentage, 2), # [cite: 855]
            'can_sit_exam': can_sit_exam # [cite: 855]
        })
    conn.close() # [cite: 855]
    return render_template('course_attendance.html', students_data=students_data, start_date=start_date, end_date=end_date) # [cite: 855]


@app.route('/students_cant_sit_exam', methods=['GET'])
def students_cant_sit_exam():
     # Your existing students_cant_sit_exam route logic from app.pdf [cite: 856, 857, 858]
    course_id = request.args.get('course_id') # [cite: 856]
    start_date = request.args.get('start_date') # [cite: 856]
    end_date = request.args.get('end_date') # [cite: 856]
    if not course_id or not start_date or not end_date: # [cite: 856]
        return "Please select a course and a valid date range.", 400 # [cite: 856]
    conn = sqlite3.connect('attendance.db') # [cite: 856]
    cursor = conn.cursor() # [cite: 857]
    cursor.execute("SELECT name FROM courses WHERE id = ?", (course_id,)) # [cite: 857]
    course_name = cursor.fetchone() # [cite: 857]
    if not course_name: # [cite: 857]
        return "Invalid course ID.", 400 # [cite: 857]
    course_name = course_name[0] # [cite: 857]
    cursor.execute('''
        SELECT s.roll_number, s.name, COUNT(a.id) AS total_classes, SUM(a.present) AS total_present
        FROM students s
        LEFT JOIN attendance a ON s.id = a.student_id AND a.course_id = ? AND a.date BETWEEN ? AND ?
        GROUP BY s.id
        ''', (course_id, start_date, end_date)) # [cite: 857]
    students_attendance = cursor.fetchall() # [cite: 857]
    students_cant_sit_exam = [] # [cite: 857]
    for student in students_attendance: # [cite: 857]
        roll_number, name, total_classes, total_present = student # [cite: 857]
        total_classes = total_classes or 0 # [cite: 857]
        total_present = total_present or 0 # [cite: 857]
        attendance_percentage = (total_present / total_classes) * 100 if total_classes > 0 else 0 # [cite: 858]
        if attendance_percentage < 50: # [cite: 858]
            students_cant_sit_exam.append({ # [cite: 858]
                'roll_number': roll_number, # [cite: 858]
                'name': name, # [cite: 858]
                'total_classes': total_classes, # [cite: 858]
                'total_present': total_present, # [cite: 858]
                'attendance_percentage': round(attendance_percentage, 2) # [cite: 858]
            })
    conn.close() # [cite: 858]
    return render_template('students_cant_sit_exam.html', course_name=course_name, students_cant_sit_exam=students_cant_sit_exam, start_date=start_date, end_date=end_date) # [cite: 858]


@app.route('/take_attendance', methods=['GET', 'POST'])
def take_attendance():
    global facial_recognition_process
    if request.method == 'GET':
        # Fetch all courses to display in the dropdown
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM courses")
        courses = cursor.fetchall()
        conn.close()
        return render_template('take_attendance.html', courses=courses)
    
    elif request.method == 'POST':
        # Get the selected course ID
        course_id = request.form.get('course_id')
        if not course_id:
            return "Please select a valid course.", 400
        
        # Check if a facial recognition process is already running
        if facial_recognition_process and facial_recognition_process.poll() is None:
            return "Facial recognition is already running. Please stop it first.", 400
        
        # Run the facial recognition script with the selected course ID
        try:
            facial_recognition_process = subprocess.Popen(['python', 'attendance_taker.py', course_id])
            return redirect(url_for('index'))  # Redirect to home page after attendance is taken
        except Exception as e:
            return f"Error running facial recognition system: {str(e)}", 500


@app.route('/stop_attendance', methods=['POST'])
def stop_attendance():
    global facial_recognition_process
    if facial_recognition_process and facial_recognition_process.poll() is None:
        # Terminate the facial recognition process
        facial_recognition_process.terminate()
        facial_recognition_process = None
        return "Facial recognition stopped successfully.", 200
    else:
        return "No facial recognition process is currently running.", 400



# --- New/Modified Routes for Face Registration ---

@app.route('/register')
def register():
    # Route to render the face registration page
    session.pop('current_folder', None) # Clear any previous session data
    session.pop('roll_number', None)
    session.pop('name', None)
    return render_template('register_face.html')

@app.route('/create_folder', methods=['POST'])
def create_folder():
    try:
        data = request.get_json()
        name = data.get('name')
        roll_number = data.get('roll_number')

        if not name or not roll_number:
            return jsonify({"status": "error", "message": "Name and Roll Number are required."}), 400

        # Simple sanitization (replace invalid chars)
        safe_name = re.sub(r'[^\w\-]+', '_', name)
        safe_roll = re.sub(r'[^\w\-]+', '_', roll_number)

        # Calculate next_person_id based on existing folders
        existing_folders = [f for f in os.listdir(FACE_IMAGES_DIR) if os.path.isdir(os.path.join(FACE_IMAGES_DIR, f)) and f.startswith("person_")] # [cite: 798]
        # Extract numbers correctly
        person_ids = []
        for folder in existing_folders:
             match = re.match(r'person_(\d+)_.*', folder)
             if match:
                  person_ids.append(int(match.group(1)))

        next_person_id = max(person_ids) + 1 if person_ids else 1 # [cite: 798]


        # Create folder name
        folder_name = f"person_{next_person_id}_roll_{safe_roll}_name_{safe_name}" # [cite: 798]
        folder_path = os.path.join(FACE_IMAGES_DIR, folder_name) # [cite: 798]

        if os.path.exists(folder_path):
             # Handle case where folder might accidentally already exist (e.g., race condition or retry)
             # Option 1: Return error
             # return jsonify({"status": "error", "message": f"Folder '{folder_name}' already exists."}), 409
             # Option 2: Allow reusing (might be simpler for user)
             pass # Allow reusing existing folder path

        else:
             os.makedirs(folder_path, exist_ok=True) # [cite: 798]
             print(f"Created folder: {folder_path}") # [cite: 630]


        # Store folder path and user details in session
        session['current_folder'] = folder_path # [cite: 798]
        session['roll_number'] = roll_number # Using original roll, not sanitized for DB
        session['name'] = name # Using original name, not sanitized for DB

        return jsonify({
            "status": "success",
            "message": f"Folder '{folder_name}' created/selected successfully.",
            "folder_path": folder_path # Send path back to JS if needed
        }) # [cite: 798]
    except Exception as e:
        print(f"Error in /create_folder: {e}") # Log error server-side
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

@app.route('/capture_image', methods=['POST'])
def capture_image():
    # Check if folder path is in session
    if 'current_folder' not in session: # [cite: 799]
        return jsonify({"status": "error", "message": "Session expired or folder not created. Please create folder again."}), 400

    folder_path = session['current_folder']

    # Verify the folder actually exists on the server
    if not os.path.isdir(folder_path):
         return jsonify({"status": "error", "message": f"Folder {folder_path} not found on server. Please create folder again."}), 404


    try:
        # Use request.form for form data
        image_data = request.form.get('image_data') # [cite: 799]
        if not image_data:
            return jsonify({"status": "error", "message": "No image data received."}), 400 # [cite: 799]

        # Decode image
        image_data_parts = image_data.split(",")
        if len(image_data_parts) != 2:
             return jsonify({"status": "error", "message": "Invalid image data format."}), 400

        image_binary = base64.b64decode(image_data_parts[1]) # [cite: 800]
        image = cv2.imdecode(np.frombuffer(image_binary, np.uint8), cv2.IMREAD_COLOR) # [cite: 800]

        if image is None:
             return jsonify({"status": "error", "message": "Could not decode image."}), 400

        # Detect faces
        # Ensure the path to the cascade file is correct
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
             return jsonify({"status": "error", "message": "Haar cascade file not found."}), 500
        face_cascade = cv2.CascadeClassifier(cascade_path) # [cite: 800]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # [cite: 800]
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # [cite: 800]

        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected in the captured image."}), 400 # [cite: 800]

        # Extract and save face ROI
        x, y, w, h = faces[0] # [cite: 800]
        padding = int(max(w, h) * 0.2) # [cite: 800]
        x_start = max(0, x - padding) # [cite: 800]
        y_start = max(0, y - padding) # [cite: 800]
        x_end = min(image.shape[1], x + w + padding) # [cite: 800]
        y_end = min(image.shape[0], y + h + padding) # [cite: 800]
        face_roi = image[y_start:y_end, x_start:x_end] # [cite: 801]

        if face_roi.size == 0:
             return jsonify({"status": "error", "message": "Failed to crop face ROI."}), 500


        resized_face = cv2.resize(face_roi, (200, 200)) # [cite: 801]

        # Save image to current folder
        img_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]) + 1 # Count existing files
        image_path = os.path.join(folder_path, f"img_face_{img_count}.jpg") # [cite: 801]

        save_success = cv2.imwrite(image_path, resized_face) # [cite: 801]

        if not save_success:
             return jsonify({"status": "error", "message": "Failed to save image file."}), 500


        return jsonify({
            "status": "success",
            "message": f"Image {img_count} saved successfully!",
            "image_count": img_count
        }) # [cite: 801]

    except Exception as e:
        print(f"Error in /capture_image: {e}") # Log error server-side
        # Provide a more generic error to the user for security
        return jsonify({"status": "error", "message": "An internal server error occurred while saving the image."}), 500


@app.route('/finalize_registration', methods=['POST'])
def finalize_registration():
    # Check if session data exists
    if 'roll_number' not in session or 'name' not in session:
        return jsonify({"status": "error", "message": "Session expired or user details not found. Please start over."}), 400

    roll_number = session['roll_number']
    name = session['name']

    try:
        # Update the database (ensure connection handling)
        conn = None
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            # Use INSERT OR IGNORE or INSERT ... ON CONFLICT to handle existing students
            cursor.execute("""
                INSERT INTO students (roll_number, name) VALUES (?, ?)
                ON CONFLICT(roll_number) DO UPDATE SET name=excluded.name
            """, (roll_number, name)) # [cite: 758]
            conn.commit()
        except sqlite3.Error as db_err:
            print(f"Database error: {db_err}")
            return jsonify({"status": "error", "message": "Database update failed."}), 500
        finally:
            if conn:
                conn.close()

        # Run features_extraction_to_csv.py
        script_path = 'features_extraction_to_csv.py'
        if not os.path.exists(script_path):
             return jsonify({"status": "error", "message": f"Script '{script_path}' not found."}), 500

        try:
            # Make sure the python executable is correct, especially in virtual envs
            # Consider using sys.executable
            result = subprocess.run(['python', script_path], check=True, capture_output=True, text=True) # [cite: 802]
            print(f"Subprocess stdout: {result.stdout}")
            print(f"Subprocess stderr: {result.stderr}")

        except FileNotFoundError:
             return jsonify({"status": "error", "message": "'python' command not found. Is Python installed and in PATH?"}), 500
        except subprocess.CalledProcessError as e:
            print(f"Subprocess error output: {e.stderr}")
            return jsonify({"status": "error", "message": f"Failed to run feature extraction script: {e.stderr}"}), 500 # [cite: 803]


        # Clear session after successful registration
        session.pop('current_folder', None)
        session.pop('roll_number', None)
        session.pop('name', None)

        return jsonify({"status": "success", "message": "Registration complete! Features updated."}) # [cite: 802]

    except Exception as e:
        print(f"Error in /finalize_registration: {e}")
        return jsonify({"status": "error", "message": f"An internal server error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True) #