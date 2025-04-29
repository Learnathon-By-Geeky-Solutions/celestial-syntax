from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
import sqlite3
import base64
import cv2
import numpy as np
import json
import subprocess
import re
import pandas as pd
from collections import defaultdict
import datetime
import sys
import functools # Import functools for the login_required decorator
import bcrypt # Import bcrypt for password hashing

SELECT_ID_FROM_STUDENTS_BY_ROLL = "SELECT id FROM students WHERE roll_number = ?"

from flask_wtf import CSRFProtect

APPLICATION_JSON = 'application/json'
TEXT_HTML = 'text/html'
LOGIN_HTML = 'login.html'

facial_recognition_process = None
app = Flask(__name__)
csrf = CSRFProtect(app)

# --- Error Handler for Database Errors ---
@app.errorhandler(sqlite3.Error)
def handle_db_error(error):
    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes[APPLICATION_JSON] > request.accept_mimetypes[TEXT_HTML]:
        return jsonify({"status": "error", "message": f"A database error occurred: {error}"}), 500
    return render_template('500.html', message="A database error occurred: {}".format(error)), 500

# --- Error Handler for Unauthorized (401) and Forbidden (403) ---
@app.errorhandler(401)
@app.errorhandler(403)
def handle_auth_error(error):
    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes[APPLICATION_JSON] > request.accept_mimetypes[TEXT_HTML]:
        return jsonify({"status": "error", "message": "Authentication required."}), error.code if hasattr(error, 'code') else 401
    return render_template(LOGIN_HTML), error.code if hasattr(error, 'code') else 401

# --- Error Handler for Internal Server Error (500) ---
@app.errorhandler(500)
def handle_internal_error(error):
    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes[APPLICATION_JSON] > request.accept_mimetypes[TEXT_HTML]:
        return jsonify({"status": "error", "message": "An internal server error occurred."}), 500
    return render_template('500.html', message="An internal server error occurred."), 500

app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24)) 

FACE_IMAGES_DIR = "data/data_faces_from_camera/"
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)
DB_NAME = 'attendance.db' # Define DB Name centrally
ATTENDANCE_SUMMARY_QUERY = "SELECT SUM(present), COUNT(*) FROM attendance"
SELECT_COURSES_QUERY = "SELECT id, name FROM courses"
SELECT_COURSE_NAME_BY_ID_QUERY = "SELECT name FROM courses WHERE id = ?"
SELECT_STUDENT_COUNT_QUERY = "SELECT COUNT(*) FROM students"
INDEX_TEMPLATE = "index.html"
LOGIN_HTML = 'login.html'
SANITIZE_REGEX = r'[^\w-]+'
REPORTS_TEMPLATE = "reports.html"
ALL_COURSES_LABEL = "All Courses"
SEMESTER_DATES = {
    "1.1": ("2022-03-01", "2022-08-31"),
    "1.2": ("2022-09-01", "2023-03-31"),
    "2.1": ("2023-04-01", "2023-08-31"), # Example dates, adjust as needed
    "2.2": ("2023-09-01", "2024-03-31"), # Example dates, adjust as needed
    "3.1": ("2024-04-01", "2024-07-31"), # Example dates, adjust as needed
    "3.2": ("2024-08-01", "2025-01-31"), # Specific dates requested by user
    # Add more semesters as needed
}

# --- Password Hashing Helpers (Using bcrypt) ---
def hash_password(password):
    """Hashes a password using bcrypt."""
    # bcrypt.gensalt() generates a salt (and work factor)
    # bcrypt.hashpw() takes the password (as bytes) and the salt (as bytes)
    # We encode the password string to bytes using UTF-8
    hashed_bytes = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    # Store the result as a UTF-8 string in the database
    return hashed_bytes.decode('utf-8')

def check_password(hashed_password_str, password_str):
    """Checks if a password matches the hashed password using bcrypt."""
    # This function is called *after* retrieving the hashed_password from DB
    if not hashed_password_str:
        return False # No stored hash to compare against (e.g., user not found)

    try:
        # bcrypt.checkpw takes the plain password (as bytes) and the stored hash (as bytes)
        # We encode both strings to bytes using UTF-8
        return bcrypt.checkpw(password_str.encode('utf-8'), hashed_password_str.encode('utf-8'))
    except ValueError:
        # This can happen if the stored hash is not a valid bcrypt hash (e.g., old format or corrupted)
        print("Warning: Stored password hash is not a valid bcrypt hash format.", file=sys.stderr)
        return False # Treat invalid hash format as incorrect password

# --- Database Helper ---
def get_db():
    """ Function to get a database connection """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    return conn


# --- Authentication Decorator ---
def login_required(view):
    """Decorator to protect routes that require authentication."""
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if 'user_id' not in session:
            # If AJAX/JSON request, return JSON error instead of redirect
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes[APPLICATION_JSON] > request.accept_mimetypes[TEXT_HTML]:
                return jsonify({"status": "error", "message": "Authentication required."}), 401
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view

# --- Login Route ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    print("--- Hit /login route ---")
    # If already logged in, redirect to index
    if 'user_id' in session:
        print("Already logged in, redirecting to index.")
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        print(f"Login attempt for username: {username}")
        conn = get_db()
        cursor = conn.cursor()
        error = None

        try:
            cursor.execute("SELECT * FROM teachers WHERE username = ?", (username,))
            teacher = cursor.fetchone()

            if teacher is None:
                error = 'Incorrect username.'
            # Use the NEW check_password function with bcrypt
            elif not check_password(teacher['password_hash'], password):
                error = 'Incorrect password.'

            if error is None:
                # Success: Log the user in
                session.clear() # Clear any old session data
                session['user_id'] = teacher['id']
                session['username'] = teacher['username'] # Store username in session too for display
                print(f"Login successful for {username}.")
                flash(f"Logged in successfully as {teacher['username']}.", "success")

                # Redirect to the next URL if stored, otherwise to index
                # next_url = session.pop('next_url', None) # Example of using stored next_url
                # return redirect(next_url or url_for('index'))
                return redirect(url_for('index'))
            else:
                print(f"Login failed for {username}: {error}")
                flash(error, "danger") # Flash error message
        except sqlite3.Error as e:
             flash(f"Database error during login: {e}", "danger")
             print(f"Login DB error: {e}", file=sys.stderr) # Log error for debugging
        except Exception as e: # Catch potential errors from check_password (like invalid hash format)
             flash("An unexpected error occurred during login.", "danger")
             print(f"Unexpected error during login: {e}", file=sys.stderr) # Log error for debugging
        finally:
            conn.close()

        # If POST fails (error occurred), re-render login page with flash messages
        return render_template(LOGIN_HTML) # Render the login template again

    # GET request: Display login form
    return render_template(LOGIN_HTML)

# --- Logout Route ---
@app.route('/logout', methods=['POST']) # Use POST for logout for security best practice
@login_required # Only logged-in users can explicitly log out
def logout():
    print("--- Hit /logout route ---")
    clear_user_session()
    print("User logged out.")
    flash("You have been logged out.", "info")
    # Redirect to the index page, which will then redirect to login if not logged in
    return redirect(url_for('index'))


# --- Apply login_required to protected routes ---

@app.route('/')
@login_required # Protect the dashboard/index page
def index():
    print("--- Hit / route (Dashboard) ---")
    conn = get_db()
    cursor = conn.cursor()

    # Fetch courses for the daily lookup dropdown
    try:
        cursor.execute(SELECT_COURSES_QUERY)
        courses = cursor.fetchall()
        print(f"Fetched {len(courses)} courses.")
    except sqlite3.Error as e:
        print(f"Database error fetching courses: {e}", file=sys.stderr)
        courses = [] # Ensure courses is an empty list on error


    # --- Calculate Overall Stats for Dashboard Card ---
    overall_percentage = 0
    total_students = 0
    try:
        # 1. Overall Attendance Percentage (All Time)
        cursor.execute(ATTENDANCE_SUMMARY_QUERY)
        overall_res = cursor.fetchone()
        overall_present = overall_res[0] if overall_res and overall_res[0] is not None else 0
        overall_total = overall_res[1] if overall_res and overall_res[1] is not None else 0
        overall_percentage = round((overall_present / overall_total) * 100, 1) if overall_total > 0 else 0
        print(f"Overall attendance: {overall_present}/{overall_total} ({overall_percentage}%)")

        # 2. Total Students Registered
        cursor.execute(SELECT_STUDENT_COUNT_QUERY)
        students_res = cursor.fetchone()
        total_students = students_res[0] if students_res and students_res[0] is not None else 0
        print(f"Total registered students: {total_students}")

    except sqlite3.Error as e:
        print(f"Database error fetching dashboard stats: {e}", file=sys.stderr) # Log error
        # Keep default values (0) if there's an error
    finally:
        conn.close() # Ensure connection is closed

    # Pass all necessary data to the index template
    print("Rendering index.html (Dashboard)")
    return render_template(INDEX_TEMPLATE,
                           selected_date='', # Needed for daily lookup form initial state
                           no_data=False, # Needed for daily lookup result display
                           courses=courses, # Needed for daily lookup form
                           semesters=SEMESTER_DATES.keys(), # Needed for student lookup form
                           student_lookup_data=None, # Initial state for student lookup result
                           course_attendance_details=None, # Initial state for student lookup result
                           no_student_data=False, # Initial state for student lookup result
                           # --- Pass dashboard stats ---
                           overall_percentage=overall_percentage,
                           total_students=total_students,
                           # --- Pass user info for display in base.html ---
                           logged_in_username=session.get('username'))


@app.route('/attendance', methods=['GET', 'POST'])
@login_required
def attendance():
    print("--- Hit /attendance route ---")
    conn = get_db()
    cursor = conn.cursor()

    courses = fetch_courses(cursor)
    overall_percentage, total_students = fetch_dashboard_stats(cursor)

    if request.method == 'GET':
        print("GET request to /attendance, redirecting to index.")
        conn.close()
        return redirect(url_for('index'))

    selected_date = request.form.get('selected_date')
    course_id = request.form.get('course_id')
    print(f"Received date: {selected_date}, course_id: {course_id} for daily lookup.")

    if not selected_date or not course_id:
        print("Missing date or course_id, flashing warning.")
        flash("Please select both a date and a course.", "warning")
        conn.close()
        return render_attendance_error(selected_date, courses, overall_percentage, total_students)

    selected_course_name = fetch_course_name(cursor, course_id)
    if selected_course_name is None:
        print(f"Course ID {course_id} not found in DB.")
        flash("Invalid course selected.", "warning")
        conn.close()
        return render_attendance_error(selected_date, courses, overall_percentage, total_students)

    try:
        attendance_data = fetch_attendance_data(cursor, selected_date, course_id)
        if not attendance_data:
            print("No attendance data found for criteria, setting no_data = True")
            flash(f"No attendance data found for {selected_course_name} on {selected_date}.", "info")
            no_data = True
        else:
            print("Attendance data found.")
            no_data = False
    except Exception as e:
        print(f"An error occurred during daily attendance fetch: {str(e)}", file=sys.stderr)
        flash(f"An error occurred fetching daily attendance: {str(e)}", "danger")
        conn.close()
        return render_attendance_error(selected_date, courses, overall_percentage, total_students)
    finally:
        conn.close()

    print("Rendering index.html with daily report results.")
    return render_template(
        INDEX_TEMPLATE,
        selected_date=selected_date,
        attendance_data=attendance_data,
        no_data=no_data,
        courses=courses,
        selected_course_name=selected_course_name,
        semesters=SEMESTER_DATES.keys(),
        student_lookup_data=None,
        course_attendance_details=None,
        no_student_data=False,
        overall_percentage=overall_percentage,
        total_students=total_students,
        logged_in_username=session.get('username')
    )


def fetch_courses(cursor):
    try:
        cursor.execute(SELECT_COURSES_QUERY)
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Database error fetching courses for attendance route: {e}", file=sys.stderr)
        return []

def fetch_dashboard_stats(cursor):
    try:
        cursor.execute(ATTENDANCE_SUMMARY_QUERY)
        overall_res = cursor.fetchone()
        overall_present = overall_res[0] if overall_res and overall_res[0] is not None else 0
        overall_total = overall_res[1] if overall_res and overall_res[1] is not None else 0
        overall_percentage = round((overall_present / overall_total) * 100, 1) if overall_total > 0 else 0
        cursor.execute(SELECT_STUDENT_COUNT_QUERY)
        students_res = cursor.fetchone()
        total_students = students_res[0] if students_res and students_res[0] is not None else 0
        return overall_percentage, total_students
    except Exception as e:
        print(f"Dashboard stats error in attendance route: {e}", file=sys.stderr)
        return 0, 0

def fetch_course_name(cursor, course_id):
    cursor.execute(SELECT_COURSE_NAME_BY_ID_QUERY, (course_id,))
    course_res = cursor.fetchone()
    if course_res:
        print(f"Found course name: {course_res['name']}")
        return course_res['name']
    return None

def fetch_attendance_data(cursor, selected_date, course_id):
    print(f"Executing query for date {selected_date}, course ID {course_id}")
    cursor.execute("""
        SELECT s.roll_number, s.name, c.name as course_name, a.present
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        JOIN courses c ON a.course_id = c.id
        WHERE a.date = ? AND a.course_id = ?
        ORDER BY s.roll_number
    """, (selected_date, course_id))
    data = cursor.fetchall()
    print(f"Query returned {len(data) if data is not None else 0} records.")
    return data

def render_attendance_error(selected_date, courses, overall_percentage, total_students):
    return render_template(
        INDEX_TEMPLATE,
        selected_date=selected_date,
        no_data=False,
        courses=courses,
        semesters=SEMESTER_DATES.keys(),
        student_lookup_data=None,
        course_attendance_details=None,
        no_student_data=False,
        overall_percentage=overall_percentage,
        total_students=total_students,
        logged_in_username=session.get('username')
    )



# --- Face Registration Routes ---
# These routes should also be protected as only teachers should register faces
@app.route('/register')
@login_required # Protect face registration
def register():
    print("--- Hit /register route ---")
    # Clear registration session data when starting registration again
    clear_registration_session()
    return render_template('register_face.html', logged_in_username=session.get('username')) # Pass username

@app.route('/create_folder', methods=['POST'])
@login_required

def create_folder():
    print("--- Hit /create_folder route ---")
    clear_registration_session()
    try:
        data = request.get_json()
        name, roll_number = data.get('name'), data.get('roll_number')
        print(f"Received name: {name}, roll_number: {roll_number}")
        error = _validate_folder_inputs(name, roll_number)
        if error:
            return error
        safe_name, safe_roll = _sanitize_inputs(name, roll_number)
        print(f"Sanitized name: {safe_name}, sanitized roll: {safe_roll}")
        is_existing_student = _check_existing_student(roll_number)
        folder_path = _get_or_create_folder(is_existing_student, safe_roll, safe_name, roll_number)
        _set_registration_session(folder_path, roll_number, name, is_existing_student)
        return jsonify({
            "status": "success",
            "message": f"Folder '{os.path.basename(folder_path)}' created/selected successfully.",
            "folder_path": folder_path
        })
    except Exception as e:
        print(f"Error in /create_folder: {e}", file=sys.stderr)
        err_str = str(e)
        if "Cannot create a file when that file already exists" in err_str or "unlink error" in err_str or "already exists" in err_str:
            return jsonify({"status": "warning", "message": f"Warning: {err_str}"}), 200
        return jsonify({"status": "error", "message": f"Server error creating folder: {err_str}"}), 500

def clear_registration_session():
    for key in ['current_folder', 'roll_number', 'name', 'is_existing_student']:
        session.pop(key, None)

def clear_user_session():
    for key in ['user_id', 'username']:
        session.pop(key, None)


def registration_session_valid():
    required = ['current_folder', 'roll_number', 'name']
    return all(k in session for k in required)

def _validate_folder_inputs(name, roll_number):
    if not name or not roll_number:
        print("Name or Roll Number missing.")
        return jsonify({"status": "error", "message": "Name and Roll Number are required."}), 400
    safe_name = re.sub(SANITIZE_REGEX, '_', name).strip('_')
    safe_roll = re.sub(SANITIZE_REGEX, '_', roll_number).strip('_')
    if not safe_name or not safe_roll:
        print("Sanitized name or roll is empty after cleaning.")
        return jsonify({"status": "error", "message": "Name or Roll Number contains invalid characters."}), 400
    return None

def _sanitize_inputs(name, roll_number):
    safe_name = re.sub(SANITIZE_REGEX, '_', name).strip('_')
    safe_roll = re.sub(SANITIZE_REGEX, '_', roll_number).strip('_')
    return safe_name, safe_roll

def _check_existing_student(roll_number):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(SELECT_ID_FROM_STUDENTS_BY_ROLL, (roll_number,))
    existing_student_row = cursor.fetchone()
    conn.close()
    is_existing_student = existing_student_row is not None
    print(f"Is existing student? {is_existing_student}")
    return is_existing_student

def _get_or_create_folder(is_existing_student, safe_roll, safe_name, roll_number):
    folder_path = None
    if is_existing_student:
        flash(f"Roll Number {roll_number} found in database. Images will be saved to the existing folder for this student.", "info")
        folder_path = _find_and_clear_existing_folder(safe_roll)
        if not folder_path:
            print(f"Warning: Roll number {roll_number} exists but no matching folder found. Creating a new one.")
            flash(f"Warning: Student with Roll Number {roll_number} exists, but their face image folder was not found. Creating a new folder.", "warning")
            is_existing_student = False
    if not folder_path:
        folder_path = _create_new_folder(safe_roll, safe_name)
    return folder_path

def _find_and_clear_existing_folder(safe_roll):
    roll_folder_pattern = re.compile(r'_roll_' + re.escape(safe_roll) + r'(_.*)?$', re.IGNORECASE)
    for folder_name in os.listdir(FACE_IMAGES_DIR):
        full_path = os.path.join(FACE_IMAGES_DIR, folder_name)
        if os.path.isdir(full_path) and roll_folder_pattern.search(folder_name):
            try:
                print(f"Clearing existing images in {full_path}")
                for f in os.listdir(full_path):
                    file_path = os.path.join(full_path, f)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                print("Existing images cleared.")
            except Exception as clear_e:
                print(f"Warning: Failed to clear existing images in {full_path}: {clear_e}", file=sys.stderr)
            return full_path
    return None

def _create_new_folder(safe_roll, safe_name):
    existing_folders = [f for f in os.listdir(FACE_IMAGES_DIR) if os.path.isdir(os.path.join(FACE_IMAGES_DIR, f)) and f.startswith("person_")]
    person_ids = []
    for folder in existing_folders:
        match = re.match(r'person_(\d+)_.*', folder)
        if match:
            person_ids.append(int(match.group(1)))
    next_person_id = max(person_ids) + 1 if person_ids else 1
    folder_name = f"person_{next_person_id}_roll_{safe_roll}_name_{safe_name}"
    folder_path = os.path.join(FACE_IMAGES_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created new folder: {folder_path}")
    return folder_path

def _set_registration_session(folder_path, roll_number, name, is_existing_student):
    session['current_folder'] = folder_path
    session['roll_number'] = roll_number
    session['name'] = name
    session['is_existing_student'] = is_existing_student
    print(f"Session data set: current_folder={session['current_folder']}, roll_number={session['roll_number']}, name={session['name']}, is_existing_student={session['is_existing_student']}")

# --- Helper Functions for Image Capture ---

def _validate_capture_session():
    """Validates registration session and folder existence."""
    if not registration_session_valid():
        flash("Session expired or registration not started...", "danger")
        clear_registration_session() # Example: Clear potentially invalid session
        return None, (jsonify({"status": "error", "message": "Session data missing."}), 400)

    folder_path = session['current_folder']
    if not os.path.isdir(folder_path):
         flash(f"Registration folder {os.path.basename(folder_path)} not found...", "danger")
         clear_registration_session()
         return None, (jsonify({"status": "error", "message": "Registration folder not found."}), 404)
    return folder_path, None # Return path if valid, None for error response

def _decode_image_from_request(request_form):
    """Decodes the base64 image data from the request."""
    image_data = request_form.get('image_data')
    if not image_data:
        return None, "No image data received."

    try:
        image_data_parts = image_data.split(",")
        if len(image_data_parts) != 2:
             return None, "Invalid image data format."
        image_binary = base64.b64decode(image_data_parts[1])
        nparr = np.frombuffer(image_binary, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
             return None, "Could not decode image."
        return image, None # Return image array if successful
    except Exception as e:
        return None, f"Error decoding image: {str(e)}"

def _find_cascade_file():
    """Finds the path to the Haar cascade file."""
    cascade_filename = 'haarcascade_frontalface_default.xml'
    base_dir = os.path.dirname(__file__)
    cascade_path = os.path.join(base_dir, cascade_filename)
    if not os.path.exists(cascade_path):
         cascade_path = os.path.join(cv2.data.haarcascades, cascade_filename)
         if not os.path.exists(cascade_path):
             print(f"Haar cascade file not found...", file=sys.stderr)
             return None
    print(f"Using cascade file: {cascade_path}")
    return cascade_path

def _detect_and_crop_face(image, cascade_path):
    """Detects a single face, crops, and resizes it."""
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(40, 40))

    if len(faces) == 0:
        return None, "No face detected in the captured image."
    if len(faces) > 1:
        print(f"Warning: Multiple faces detected ({len(faces)}). Using the first.")

    x, y, w, h = faces[0]
    padding = int(max(w, h) * 0.3)
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(image.shape[1], x + w + padding)
    y_end = min(image.shape[0], y + h + padding)
    face_roi = image[y_start:y_end, x_start:x_end]

    if face_roi.size == 0:
         return None, "Failed to crop face region (ROI empty)."

    resized_face = cv2.resize(face_roi, (200, 200))
    if resized_face.size == 0:
        return None, "Failed to crop face region (resize failed)."

    return resized_face, None

def _save_face_image(folder_path, face_image):
    """Saves the face image to the specified folder."""
    try:
        img_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]) + 1
        image_path = os.path.join(folder_path, f"img_face_{img_count}.jpg")
        print(f"Saving image to: {image_path}")
        save_success = cv2.imwrite(image_path, face_image)
        if not save_success:
             return -1, "Failed to save image file." # Return -1 for error count
        return img_count, None
    except Exception as e:
        return -1, f"Error saving image: {str(e)}"


# --- Refactored Route ---

@app.route('/capture_image', methods=['POST'])
@login_required
def capture_image():
    print("--- Hit /capture_image route ---")

    # 1. Validate Session and Folder
    folder_path, error_response = _validate_capture_session()
    if error_response:
        return error_response # Returns the jsonify response directly

    # 2. Decode Image
    image, error_msg = _decode_image_from_request(request.form)
    if error_msg:
        print(f"Image decode error: {error_msg}")
        return jsonify({"status": "error", "message": error_msg}), 400

    # 3. Load Cascade
    cascade_path = _find_cascade_file()
    if not cascade_path:
        flash("Facial recognition cascade file not found on server.", "danger")
        return jsonify({"status": "error", "message": "Facial recognition configuration error."}), 500

    # --- Wrapped Face Processing in try/except ---
    # This reduces nesting inside the main try block of the original function
    try:
        # 4. Detect and Crop Face
        resized_face, error_msg = _detect_and_crop_face(image, cascade_path)
        if error_msg:
             # Use 400 for detection issues, 500 for cropping/resizing issues
             status_code = 400 if "No face detected" in error_msg else 500
             flash(error_msg, "warning" if status_code == 400 else "danger")
             print(f"Face processing error: {error_msg}")
             return jsonify({"status": "error", "message": error_msg}), status_code

        # 5. Save Image
        img_count, error_msg = _save_face_image(folder_path, resized_face)
        if error_msg:
             print(f"Image save error: {error_msg}", file=sys.stderr)
             flash(f"Failed to save image: {error_msg}", "danger")
             return jsonify({"status": "error", "message": "Failed to save image file."}), 500

        # 6. Return Success
        print(f"Image {img_count} saved successfully.")
        return jsonify({
            "status": "success",
            "message": f"Image {img_count} saved successfully!",
            "image_count": img_count
        }), 200

    except Exception as e:
        # Catch unexpected errors specifically during OpenCV processing or saving
        print(f"Unexpected error during face processing/saving: {e}", file=sys.stderr)
        flash(f"An internal server error occurred during image capture: {str(e)}", "danger")
        return jsonify({"status": "error", "message": f"An internal server error occurred: {str(e)}"}), 500

@app.route('/finalize_registration', methods=['POST'])
@login_required
def finalize_registration():
    print("--- Hit /finalize_registration route ---")
    # Validate session
    valid, session_data_or_resp = validate_registration_session()
    if not valid:
        return session_data_or_resp
    roll_number, name, folder_path, is_existing_student = session_data_or_resp

    # Check for images
    valid, image_resp = check_images_for_finalization(folder_path)
    if not valid:
        return image_resp

    # Insert or update student
    db_success, db_resp = insert_or_update_student(roll_number, name, is_existing_student)
    if not db_success:
        return db_resp

    # Feature extraction
    run_feature_extraction()

    # Cleanup session
    cleanup_registration_session()
    print("Session data cleared after finalization.")

    return jsonify({"status": "success", "message": "Registration finalized."})


def validate_registration_session():
    if not registration_session_valid():
        flash("Session expired or registration not started. Please start registration over.", "danger")
        clear_registration_session()
        print("Session data missing for finalization.")
        return False, (jsonify({"status": "error", "message": "Session data missing for finalization."}), 400)
    roll_number = session['roll_number']
    name = session['name']
    folder_path = session['current_folder']
    is_existing_student = session.get('is_existing_student', False)
    print(f"Finalizing registration for roll: {roll_number}, name: {name}, folder: {folder_path}, existing: {is_existing_student}")
    return True, (roll_number, name, folder_path, is_existing_student)

def check_images_for_finalization(folder_path):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        flash(f"No face images were successfully saved in the folder '{os.path.basename(folder_path)}'. Please capture images before finalizing.", "warning")
        print(f"No images found in folder {folder_path} for finalization.")
        return False, (jsonify({"status": "error", "message": "No images saved for this registration."}), 400)
    return True, None

def insert_or_update_student(roll_number, name, is_existing_student):
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor()
        if not is_existing_student:
            print(f"Attempting to insert new student: {roll_number}, {name}")
            cursor.execute(SELECT_ID_FROM_STUDENTS_BY_ROLL, (roll_number,))
            existing_student_check = cursor.fetchone()
            if existing_student_check:
                flash(f"Roll Number {roll_number} already exists in the database.", "danger")
                if conn: conn.rollback()
                cleanup_registration_session()
                print(f"Concurrency error: Roll number {roll_number} found during finalization insert.")
                return False, (jsonify({"status": "error", "message": f"Roll Number {roll_number} already exists."}), 500)
            cursor.execute("INSERT INTO students (roll_number, name) VALUES (?, ?)", (roll_number, name))
            conn.commit()
            print(f"Successfully inserted student {roll_number}.")
        else:
            print(f"Updating existing student name for {roll_number} to {name}")
            cursor.execute("UPDATE students SET name = ? WHERE roll_number = ?", (name, roll_number))
            conn.commit()
            print(f"Successfully updated student {roll_number}.")
        return True, None
    except sqlite3.Error as db_err:
        print(f"Database error saving student during finalization: {db_err}", file=sys.stderr)
        flash(f"Database error occurred during student save: {db_err}", "danger")
        if conn: conn.rollback()
        cleanup_registration_session()
        return False, (jsonify({"status": "error", "message": "Database error saving student data."}), 500)
    finally:
        if conn: conn.close()

def run_feature_extraction():
    script_path = os.path.join(os.path.dirname(__file__), 'features_extraction_to_csv.py')
    print(f"Checking for feature extraction script at: {script_path}")
    if not os.path.exists(script_path):
        print(f"Feature extraction script not found at: {script_path}", file=sys.stderr)
        flash("Feature extraction script not found on the server.", "danger")
        return
    try:
        python_executable = sys.executable
        print(f"Running feature extraction: {python_executable} {script_path}")
        result = subprocess.run([python_executable, script_path], check=True, capture_output=True, text=True, cwd=os.path.dirname(__file__), encoding='utf-8')
        print(f"Feature Extraction stdout:\n{result.stdout}")
        if result.stderr:
            print(f"Feature Extraction stderr:\n{result.stderr}", file=sys.stderr)
        flash("Registration complete! Student data saved and face features updated.", "success")
        print("Feature extraction completed successfully (or with warnings).")
    except FileNotFoundError:
        print(f"Python executable not found: {python_executable}", file=sys.stderr)
        flash("Error: Python command not found on the server. Feature extraction failed.", "danger")
    except subprocess.CalledProcessError as e:
        print(f"Feature Extraction script failed. Return code: {e.returncode}", file=sys.stderr)
        print(f"Feature Extraction error output:\n{e.stderr}", file=sys.stderr)
        flash("Feature extraction script failed. See server logs for details.", "danger")
    except Exception as e:
        print(f"Unexpected error running feature extraction: {e}", file=sys.stderr)
        flash(f"An unexpected error occurred running feature extraction: {str(e)}", "danger")

def cleanup_registration_session():
    session.pop('current_folder', None)
    session.pop('roll_number', None)
    session.pop('name', None)
    session.pop('is_existing_student', None)


# Add these helper functions before the reports() route

def _get_filter_dates(selected_semester):
    """Determines start and end dates based on the selected semester."""
    start_date, end_date = None, None
    if selected_semester != 'all' and selected_semester in SEMESTER_DATES:
        start_date, end_date = SEMESTER_DATES[selected_semester]
        print(f"Applying semester filter: {start_date} to {end_date}")
    else:
        print("No semester filter applied.")
    return start_date, end_date

def _build_date_filter_sql(start_date, end_date):
    """Builds the WHERE clause and parameters for date filtering."""
    date_filter_sql = ""
    query_params = []
    if start_date and end_date:
        date_filter_sql = " WHERE a.date BETWEEN ? AND ? "
        query_params = [start_date, end_date]
    return date_filter_sql, query_params

def _fetch_all_courses_for_report(cursor):
    """Fetches all courses for the report dropdown."""
    try:
        cursor.execute("SELECT id, name FROM courses ORDER BY name")
        courses = cursor.fetchall()
        print(f"Fetched {len(courses)} courses for reports.")
        return courses
    except sqlite3.Error as e:
        print(f"Database error fetching courses for report: {e}", file=sys.stderr)
        return [] # Return empty list on error

def _calculate_overall_attendance(cursor, date_filter_sql, query_params):
    """Calculates the overall attendance percentage within the filtered dates."""
    overall_sql = f"SELECT SUM(present), COUNT(*) FROM attendance a {date_filter_sql.replace(' WHERE', 'WHERE') if date_filter_sql else ''}"
    print(f"Executing overall SQL: {overall_sql} with params {query_params}")
    cursor.execute(overall_sql, query_params)
    overall_res = cursor.fetchone()
    overall_present = overall_res[0] if overall_res and overall_res[0] is not None else 0
    overall_total = overall_res[1] if overall_res and overall_res[1] is not None else 0
    percentage = (overall_present / overall_total) * 100 if overall_total > 0 else 0
    print(f"Overall filtered attendance: {overall_present}/{overall_total} ({round(percentage, 2)}%)")
    return round(percentage, 1)

def _calculate_course_stats(cursor, date_filter_sql, query_params):
    """Calculates attendance statistics for each course within the filtered dates."""
    course_sql = f"""
        SELECT
            c.id, c.name, SUM(a.present) AS total_present, COUNT(a.id) AS total_records
        FROM attendance a
        JOIN courses c ON a.course_id = c.id
        {date_filter_sql.replace(' WHERE', 'WHERE') if date_filter_sql else ''}
        GROUP BY c.id, c.name
        ORDER BY c.name
    """
    print(f"Executing course stats SQL: {course_sql} with params {query_params}")
    cursor.execute(course_sql, query_params)
    course_stats_raw = cursor.fetchall()
    course_stats = []
    for row in course_stats_raw:
        total_present = row['total_present'] if row['total_present'] is not None else 0
        total_records = row['total_records'] if row['total_records'] is not None else 0
        percentage = (total_present / total_records) * 100 if total_records > 0 else 0
        course_stats.append({
            'id': row['id'], 'name': row['name'], 'percentage': round(percentage, 2),
            'present': total_present, 'total': total_records
        })
    print(f"Fetched {len(course_stats)} course stats.")
    return course_stats

def _calculate_attendance_trend(cursor, date_filter_sql, query_params):
    """Calculates the monthly attendance trend within the filtered dates."""
    trend_sql = f"""
        SELECT strftime('%Y-%m', date) AS month, SUM(present) AS monthly_present, COUNT(id) AS monthly_total
        FROM attendance a
        {date_filter_sql.replace(' WHERE', 'WHERE') if date_filter_sql else ''}
        GROUP BY month ORDER BY month
    """
    print(f"Executing trend SQL: {trend_sql} with params {query_params}")
    cursor.execute(trend_sql, query_params)
    trend_raw = cursor.fetchall()
    attendance_trend = {'labels': [], 'percentages': [], 'present_counts': [], 'total_counts': []}
    for row in trend_raw:
        monthly_present = row['monthly_present'] or 0
        monthly_total = row['monthly_total'] or 0
        percentage = (monthly_present / monthly_total) * 100 if monthly_total > 0 else 0
        attendance_trend['labels'].append(row['month'])
        attendance_trend['percentages'].append(round(percentage, 2))
        attendance_trend['present_counts'].append(monthly_present)
        attendance_trend['total_counts'].append(monthly_total)
    print("Fetched attendance trend data for reports.")
    return attendance_trend

def _validate_course_filter(cursor, selected_course_id):
    """Validates the selected course ID and fetches its name."""
    course_id_int = None
    selected_course_name = ALL_COURSES_LABEL # Default
    is_valid = False

    if selected_course_id != 'all':
        try:
            course_id_int = int(selected_course_id)
            cursor.execute(SELECT_COURSE_NAME_BY_ID_QUERY, (course_id_int,))
            course_name_res = cursor.fetchone()
            if course_name_res:
                selected_course_name = course_name_res['name']
                is_valid = True
                print(f"Applying course filter: ID={course_id_int}, Name={selected_course_name}")
            else:
                flash(f"Invalid course ID ({selected_course_id}) provided for filtering.", "warning")
                print(f"Invalid course ID {selected_course_id} provided for filtering low attendance.")
        except ValueError:
            flash(f"Invalid course ID format ('{selected_course_id}') provided.", "warning")
            print(f"Invalid course ID format '{selected_course_id}' provided for filtering low attendance.")
        except sqlite3.Error as e:
             flash(f"Database error validating course ID: {e}", "danger")
             print(f"DB error validating course ID {selected_course_id}: {e}", file=sys.stderr)
    else:
         is_valid = True # 'all' is always valid

    # Return the integer ID only if valid and not 'all', otherwise None
    return course_id_int if is_valid and selected_course_id != 'all' else None, selected_course_name, is_valid


def _calculate_low_attendance_students(cursor, start_date, end_date, valid_course_id_int, low_threshold):
    """Calculates the list of students below the low attendance threshold, applying filters."""
    where_clauses = []
    query_params = []

    if start_date and end_date:
        where_clauses.append("a.date BETWEEN ? AND ?")
        query_params.extend([start_date, end_date])

    if valid_course_id_int is not None: # Use the validated integer ID
        where_clauses.append("a.course_id = ?")
        query_params.append(valid_course_id_int)

    where_sql = ""
    if where_clauses:
        where_sql = " WHERE " + " AND ".join(where_clauses)

    low_attendance_sql = f"""
        SELECT
            s.id, s.roll_number, s.name,
            SUM(CASE WHEN a.present = 1 THEN 1 ELSE 0 END) AS student_present_period,
            COUNT(a.id) AS student_total_period
        FROM students s
        JOIN attendance a ON s.id = a.student_id
        {where_sql}
        GROUP BY s.id, s.roll_number, s.name
        HAVING COUNT(a.id) > 0
    """
    print(f"Executing low attendance SQL: {low_attendance_sql} with params {query_params}")
    cursor.execute(low_attendance_sql, query_params)
    period_student_stats = cursor.fetchall()

    low_attendance_students = []
    for student in period_student_stats:
        student_present = student['student_present_period'] or 0
        student_total = student['student_total_period'] or 0
        percentage = (student_present / student_total) * 100 if student_total > 0 else 0
        if percentage < low_threshold:
            low_attendance_students.append({
                'roll_number': student['roll_number'], 'name': student['name'],
                'percentage': round(percentage, 2),
                'present': student_present, 'total': student_total
            })

    low_attendance_students.sort(key=lambda x: x['percentage'])
    print(f"Found {len(low_attendance_students)} students below {low_threshold}% attendance.")
    return low_attendance_students

def _format_date_range_display(start_date, end_date):
    """Formats the date range string for display."""
    if start_date and end_date:
        try:
            # Attempt to parse and reformat dates to DD-MM-YYYY
            start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            start_display = start_dt.strftime('%d-%m-%Y')
            end_display = end_dt.strftime('%d-%m-%Y')
            return f"({start_display} to {end_display})"
        except ValueError:
            # Fallback if parsing fails (shouldn't happen with YYYY-MM-DD)
             return f"({start_date} to {end_date})" # Show original format
    return "(All Time)"


# --- UPDATED Reporting and Analytics Route ---
@app.route('/reports')
@login_required # Protect the reports page
def reports():
    low_threshold = 60  # Example threshold percentage
    print("--- Hit /reports route ---")
    conn = None
    all_courses = []
    overall_percentage = 0
    course_stats = []
    attendance_trend = {'labels': [], 'percentages': [], 'present_counts': [], 'total_counts': []}
    low_attendance_students = []
    error_message = None
    selected_course_name = ALL_COURSES_LABEL
    is_course_filter_valid = False
    date_range_display = ""

    # Get filters from request arguments
    selected_semester = request.args.get('semester', 'all')
    # Use str() to handle None or numeric course_id safely
    selected_course_id_str = str(request.args.get('course_id', 'all'))
    print(f"Reports filter: semester={selected_semester}, course_id='{selected_course_id_str}'")

    try:
        conn = get_db()
        cursor = conn.cursor()

        # Fetch data common to all reports
        all_courses = _fetch_all_courses_for_report(cursor)

        # Determine date range based on semester
        start_date, end_date = _get_filter_dates(selected_semester)
        date_filter_sql, date_query_params = _build_date_filter_sql(start_date, end_date)

        # Calculate report sections using helpers
        overall_percentage = _calculate_overall_attendance(cursor, date_filter_sql, date_query_params)
        course_stats = _calculate_course_stats(cursor, date_filter_sql, date_query_params)
        attendance_trend = _calculate_attendance_trend(cursor, date_filter_sql, date_query_params)

        # Validate the course filter specifically for the low attendance list
        # This keeps the validation logic close to where it's needed
        valid_course_id_int, selected_course_name, is_course_filter_valid = _validate_course_filter(cursor, selected_course_id_str)

        # Only calculate low attendance if the course filter was valid (either 'all' or a real course ID)
        if is_course_filter_valid:
             low_attendance_students = _calculate_low_attendance_students(
                 cursor, start_date, end_date, valid_course_id_int, low_threshold
             )
        else:
             # If course filter was invalid, don't attempt calculation, maybe add a note
             print("Skipping low attendance calculation due to invalid course filter.")
             # Flash message is handled within _validate_course_filter

        # Format date range for display
        date_range_display = _format_date_range_display(start_date, end_date)

        # --- IMPORTANT: Remove the duplicated low attendance block that was here ---
        # The logic from lines ~1007 to ~1056 in the original file should be deleted.

    except sqlite3.Error as e:
        print(f"Database error in /reports route: {e}", file=sys.stderr)
        error_message = f"A database error occurred: {e}"
        flash(error_message, "danger")
    except Exception as e:
        print(f"Unexpected error in /reports route: {e}", file=sys.stderr)
        error_message = f"An unexpected error occurred: {e}"
        flash(error_message, "danger")
        # Consider re-raising or logging traceback here for deeper debugging if needed
        # import traceback
        # traceback.print_exc()
    finally:
        if conn:
            conn.close()

    # Render the template with fetched/calculated data or defaults in case of error
    print("Rendering reports.html")
    return render_template(REPORTS_TEMPLATE,
                           overall_percentage=overall_percentage, # Already rounded in helper
                           course_stats=course_stats,
                           attendance_trend=attendance_trend,
                           low_attendance_students=low_attendance_students,
                           low_threshold=low_threshold,
                           semesters=SEMESTER_DATES.keys(),
                           selected_semester=selected_semester,
                           all_courses=all_courses,
                           selected_course_id=selected_course_id_str, # Pass the original string back for form default
                           selected_course_name=selected_course_name if is_course_filter_valid else ALL_COURSES_LABEL, # Use validated name
                           date_range_display=date_range_display,
                           logged_in_username=session.get('username'),
                           error_message=error_message) # Pass error message if any


def render_student_lookup_page(courses_for_daily_lookup, overall_percentage, total_students,
                               student_data=None, course_details=None, no_student_data=False,
                               selected_roll='', selected_semester=''):
    """Renders the index.html template with student lookup results."""
    print("Rendering index.html with student semester lookup results.")
    return render_template(
        INDEX_TEMPLATE,
        selected_date='', # For daily lookup part
        no_data=False, # For daily lookup part
        courses=courses_for_daily_lookup,
        semesters=SEMESTER_DATES.keys(),
        student_lookup_data=student_data,
        course_attendance_details=course_details if course_details is not None else [],
        no_student_data=no_student_data,
        overall_percentage=overall_percentage,
        total_students=total_students,
        logged_in_username=session.get('username'),
        # Pass back form selections for persistence
        lookup_roll_number=selected_roll,
        lookup_semester=selected_semester
    )

# --- Helper function to process student attendance data ---
def process_student_attendance_for_semester(cursor, roll_number, selected_semester):
    """Fetches and processes attendance data for a student in a given semester."""
    student_data = None
    course_attendance_details = []
    no_student_data = False
    error_message = None # To capture specific errors

    # 1. Validate Semester and Get Dates
    if selected_semester not in SEMESTER_DATES:
        return None, [], True, "Invalid semester selected." # Return error message

    start_date, end_date = SEMESTER_DATES[selected_semester]
    print(f"Semester dates for {selected_semester}: {start_date} to {end_date}")

    # 2. Fetch Student
    student_res = fetch_student_by_roll(cursor, roll_number)
    if not student_res:
        print(f"No student found with Roll Number: {roll_number}")
        return None, [], True, f"No student found with Roll Number: {roll_number}" # Return error message

    student_id = student_res['id']
    student_name = student_res['name']
    student_data = {'roll_number': roll_number, 'name': student_name, 'semester': selected_semester}
    print(f"Found student: {student_name} (ID: {student_id})")

    # 3. Fetch Courses with Sessions in Semester
    all_semester_courses = fetch_courses_in_semester(cursor, start_date, end_date)
    print(f"Found {len(all_semester_courses)} courses with sessions in semester {selected_semester}.")
    if not all_semester_courses:
        # Student exists, but no classes were held in this semester for *any* course.
        # It's still valid data, just nothing to show attendance for.
        print("No classes recorded in this semester for any course.")
        # We don't set no_student_data=True here, because the student *was* found.
        # We return the student data but empty course details.
        # A specific message might be flashed later.
        return student_data, [], False, f"No classes were recorded for any course during semester {selected_semester}."


    # 4. Calculate Attendance for Each Course
    has_records_for_student = False
    for course in all_semester_courses:
        course_id = course['id']
        course_name = course['name']
        print(f"Calculating attendance for course: {course_name}")
        total_classes_held, total_present, total_absent = fetch_attendance_counts(
            cursor, student_id, course_id, start_date, end_date)
        print(f" - Total classes held: {total_classes_held}")
        print(f" - Classes attended: {total_present}")
        print(f" - Classes absent: {total_absent}")

        # Only add details if classes were actually held for this course in the semester
        if total_classes_held > 0:
            percentage = (total_present / total_classes_held) * 100 if total_classes_held > 0 else 0
            print(f" - Percentage: {round(percentage, 2)}%")
            course_attendance_details.append({
                'course_name': course_name,
                'total_classes': total_classes_held,
                'classes_attended': total_present,
                'classes_absent': total_absent,
                'percentage': round(percentage, 2)
            })
            # Check if there was at least one record (present or absent) for *this* student
            if total_present > 0 or total_absent > 0:
                 has_records_for_student = True

    # 5. Check if Student Found but No Attendance Records for *Them*
    if not has_records_for_student and all_semester_courses:
         # Student exists, courses had sessions, but this student has 0 records in those sessions.
         print("Student found, but no attendance records in this semester/course.")
         # We still return student data and empty course details, maybe flash a message later.
         error_message = f"No attendance records found for {student_name} ({roll_number}) in any course within semester {selected_semester}."
         # Setting no_student_data=True might be confusing as the student *was* found.
         # It's better to show the student info and an empty table with a message.

    return student_data, course_attendance_details, False, error_message # Return False for no_student_data if student was found


@app.route('/student_semester_attendance', methods=['POST'])
@login_required # Protect student lookup
def student_semester_attendance():
    print("--- Hit /student_semester_attendance route ---")
    roll_number = request.form.get('roll_number')
    selected_semester = request.form.get('semester')

    # Initialize variables for template rendering
    student_data = None
    course_attendance_details = []
    no_student_data = False # Flag specifically for "student not found" or major error

    conn = None # Initialize conn to None
    courses_for_daily_lookup = []
    overall_percentage = 0
    total_students = 0

    try:
        conn = get_db()
        cursor = conn.cursor()

        # Fetch data needed for the page regardless of lookup result
        courses_for_daily_lookup = fetch_courses_for_lookup(cursor)
        overall_percentage, total_students = fetch_dashboard_stats_for_lookup(cursor) # Use renamed helper

        # 1. Validate Input (using existing helper)
        valid, error_msg = validate_student_lookup_input(roll_number, selected_semester)
        if not valid:
            print(f"Input validation failed: {error_msg}")
            flash(error_msg, "warning")
            # Render the page with the error message, preserving input if possible
            return render_student_lookup_page(
                courses_for_daily_lookup, overall_percentage, total_students,
                no_student_data=True, # Indicate lookup failed due to input
                selected_roll=roll_number, selected_semester=selected_semester
            )

        # 2. Process Attendance Data using the new helper function
        student_data, course_attendance_details, no_student_data, message = process_student_attendance_for_semester(
            cursor, roll_number, selected_semester
        )

        # 3. Flash appropriate message based on processing results
        if message:
            if no_student_data:  # This means student wasn't found or semester was invalid
                flash(f"Lookup failed: {message}", "info")
            else:
                # Student was found, but maybe no classes or no records for them
                flash(f"Note: {message} (student found, but no attendance records)", "warning")


    except sqlite3.Error as e:
        flash(f"Database error occurred during student semester lookup: {e}", "danger")
        print(f"Student Lookup DB error: {e}", file=sys.stderr)
        no_student_data = True # Indicate a major error prevented lookup
        student_data = None # Clear any partial data
        course_attendance_details = []
    except Exception as e: # Catch unexpected errors
        flash(f"An unexpected error occurred: {e}", "danger")
        print(f"Unexpected error in student lookup: {e}", file=sys.stderr)
        no_student_data = True
        student_data = None
        course_attendance_details = []
    finally:
        if conn:
            conn.close()

    # 4. Render the page with the results
    return render_student_lookup_page(
        courses_for_daily_lookup, overall_percentage, total_students,
        student_data=student_data,
        course_details=course_attendance_details,
        no_student_data=no_student_data, # Pass the final flag
        selected_roll=roll_number, # Preserve form selection
        selected_semester=selected_semester
    )



# --- Helper functions for student_semester_attendance ---
def fetch_courses_for_lookup(cursor):
    try:
        cursor.execute(SELECT_COURSES_QUERY)
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Database error fetching courses for student lookup route: {e}", file=sys.stderr)
        return []

def fetch_dashboard_stats_for_lookup(cursor):
    overall_percentage = 0
    total_students = 0
    try:
        cursor.execute(ATTENDANCE_SUMMARY_QUERY)
        overall_res = cursor.fetchone()
        overall_present = overall_res[0] if overall_res and overall_res[0] is not None else 0
        overall_total = overall_res[1] if overall_res and overall_res[1] is not None else 0
        overall_percentage = round((overall_present / overall_total) * 100, 1) if overall_total > 0 else 0

        cursor.execute(SELECT_STUDENT_COUNT_QUERY)
        students_res = cursor.fetchone()
        total_students = students_res[0] if students_res and students_res[0] is not None else 0
    except Exception as e:
        print(f"Dashboard stats error in student lookup route: {e}", file=sys.stderr)
    return overall_percentage, total_students

def validate_student_lookup_input(roll_number, selected_semester):
    if not roll_number or not selected_semester:
        return False, "Please enter Roll Number and select a Semester."
    if selected_semester not in SEMESTER_DATES:
        return False, "Invalid semester selected."
    return True, None

def fetch_student_by_roll(cursor, roll_number):
    cursor.execute("SELECT id, name FROM students WHERE roll_number = ?", (roll_number,))
    return cursor.fetchone()

def fetch_courses_in_semester(cursor, start_date, end_date):
    cursor.execute("""
        SELECT DISTINCT c.id, c.name
        FROM attendance a
        JOIN courses c ON a.course_id = c.id
        WHERE a.date BETWEEN ? AND ?
        ORDER BY c.name
    """, (start_date, end_date))
    return cursor.fetchall()

def fetch_attendance_counts(cursor, student_id, course_id, start_date, end_date):
    # Total classes held
    cursor.execute("""
        SELECT COUNT(DISTINCT date)
        FROM attendance
        WHERE course_id = ? AND date BETWEEN ? AND ?
    """, (course_id, start_date, end_date))
    total_classes_held_res = cursor.fetchone()
    total_classes_held = total_classes_held_res[0] if total_classes_held_res and total_classes_held_res[0] is not None else 0

    # Classes attended
    cursor.execute("""
        SELECT COUNT(*)
        FROM attendance
        WHERE student_id = ? AND course_id = ? AND present = 1 AND date BETWEEN ? AND ?
    """, (student_id, course_id, start_date, end_date))
    total_present_res = cursor.fetchone()
    total_present = total_present_res[0] if total_present_res and total_present_res[0] is not None else 0

    # Classes absent
    cursor.execute("""
        SELECT COUNT(*)
        FROM attendance
        WHERE student_id = ? AND course_id = ? AND present = 0 AND date BETWEEN ? AND ?
    """, (student_id, course_id, start_date, end_date))
    total_absent_res = cursor.fetchone()
    total_absent = total_absent_res[0] if total_absent_res and total_absent_res[0] is not None else 0

    return total_classes_held, total_present, total_absent


@app.route('/take_attendance', methods=['GET'])
@login_required
def take_attendance():
    # Fetch all courses to display in the dropdown
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute(SELECT_COURSES_QUERY)
        courses = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Database error fetching courses for take attendance: {e}", file=sys.stderr)
        courses = []
    finally:
        conn.close()
    return render_template('take_attendance.html', courses=courses, logged_in_username=session.get('username'))

# --- API endpoint for browser-based face recognition ---
from io import BytesIO
from PIL import Image
import re

@app.route('/api/recognize_face', methods=['POST'])
@login_required
@csrf.exempt
def api_recognize_face():
    data = request.get_json()
    img_data = data.get('image')
    course_id = data.get('course_id')
    if not img_data or not course_id:
        return jsonify({'status': 'error', 'message': 'Missing image or course_id'}), 400
    
    img_bgr, err = parse_base64_image(img_data)
    if err:
        return jsonify({'status': 'error', 'message': f'Invalid image data: {err}'}), 400
    
    try:
        recognized_people, face_error = recognize_faces_and_mark_attendance(img_bgr, course_id)
        if face_error:
            return jsonify({'status': 'error', 'message': face_error}), 500
        if not recognized_people:
            return jsonify({'status': 'success', 'recognized': []})
        return jsonify({'status': 'success', 'recognized': recognized_people})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Recognition error: {e}'}), 500


def parse_base64_image(img_data):
    import re, base64
    from PIL import Image
    from io import BytesIO
    try:
        img_str = re.sub('^data:image/.+;base64,', '', img_data)
        img_bytes = base64.b64decode(img_str)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_bgr, None
    except Exception as e:
        return None, str(e)


def recognize_faces_and_mark_attendance(img_bgr, course_id):
    try:
        from attendance_taker import Face_Recognizer
        recognizer = Face_Recognizer(course_id)
        recognizer.get_face_database()
        detector = getattr(recognizer, 'detector', None)
        if detector is None:
            import dlib
            detector = dlib.get_frontal_face_detector()
        faces = detector(img_bgr, 1)
        if len(faces) == 0:
            return [], None
        predictor = getattr(recognizer, 'predictor', None)
        face_reco_model = getattr(recognizer, 'face_reco_model', None)
        if predictor is None or face_reco_model is None:
            import dlib
            predictor = dlib.shape_predictor(r"data/data_dlib/shape_predictor_68_face_landmarks.dat")
            face_reco_model = dlib.face_recognition_model_v1(r"data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
        recognized_people = []
        for face in faces:
            shape = predictor(img_bgr, face)
            features = face_reco_model.compute_face_descriptor(img_bgr, shape)
            min_dist, min_idx = float('inf'), -1
            for i, feat_known in enumerate(recognizer.face_features_known_list):
                dist = recognizer.return_euclidean_distance(features, feat_known)
                if dist < min_dist:
                    min_dist, min_idx = dist, i
            if min_dist < 0.4 and min_idx >= 0:
                roll = recognizer.face_roll_number_known_list[min_idx]
                name = recognizer.face_name_known_list[min_idx]
                recognized_people.append({'name': name, 'roll': roll})
                mark_attendance_for_student(roll, course_id)
            else:
                recognized_people.append({'name': 'Unknown', 'roll': '-'})
        return recognized_people, None
    except Exception as e:
        return [], str(e)


def mark_attendance_for_student(roll, course_id):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        cursor.execute(SELECT_ID_FROM_STUDENTS_BY_ROLL, (roll,))
        student = cursor.fetchone()
        if student:
            student_id = student[0]
            cursor.execute("""
                UPDATE attendance SET present=1 WHERE student_id=? AND course_id=? AND date=?
            """, (student_id, course_id, today))
            if cursor.rowcount == 0:
                cursor.execute("""
                    INSERT INTO attendance (student_id, course_id, date, present) VALUES (?, ?, ?, 1)
                """, (student_id, course_id, today))
            conn.commit()
        conn.close()
    except Exception as db_e:
        print(f"[ERROR] Attendance DB update: {db_e}")
if __name__ == '__main__':

    app.run()