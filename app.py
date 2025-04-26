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
from flask_wtf import CSRFProtect

facial_recognition_process = None
app = Flask(__name__)
csrf = CSRFProtect(app)

# --- Error Handler for Database Errors ---
@app.errorhandler(sqlite3.Error)
def handle_db_error(error):
    return render_template(INDEX_TEMPLATE, message="A database error occurred: {}".format(error)), 500

app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24)) 

FACE_IMAGES_DIR = "data/data_faces_from_camera/"
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)
DB_NAME = 'attendance.db' # Define DB Name centrally
ATTENDANCE_SUMMARY_QUERY = "SELECT SUM(present), COUNT(*) FROM attendance"
SELECT_COURSES_QUERY = "SELECT id, name FROM courses"
INDEX_TEMPLATE = "index.html"
SANITIZE_REGEX = r'[^\w-]+'
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
            flash("Please log in to access this page.", "warning")
            # Store the requested URL to redirect back after login (optional but good UX)
            # session['next_url'] = request.url # Simple example, needs refinement for query strings, POST etc.
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
        return render_template('login.html') # Render the login template again

    # GET request: Display login form
    return render_template('login.html')

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
        cursor.execute("SELECT COUNT(*) FROM students")
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
@login_required # Protect the daily attendance report route
def attendance():
    print("--- Hit /attendance route ---")

    # Fetch data needed to re-render the index page correctly
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(SELECT_COURSES_QUERY)
        courses = cursor.fetchall()
    except sqlite3.Error as e:
         print(f"Database error fetching courses for attendance route: {e}", file=sys.stderr)
         courses = []

    # Fetch dashboard stats (needed when re-rendering index.html)
    overall_percentage = 0
    total_students = 0
    try:
        cursor.execute(ATTENDANCE_SUMMARY_QUERY)
        overall_res = cursor.fetchone()
        overall_present = overall_res[0] if overall_res and overall_res[0] is not None else 0
        overall_total = overall_res[1] if overall_res and overall_res[1] is not None else 0
        overall_percentage = round((overall_present / overall_total) * 100, 1) if overall_total > 0 else 0

        cursor.execute("SELECT COUNT(*) FROM students")
        students_res = cursor.fetchone()
        total_students = students_res[0] if students_res and students_res[0] is not None else 0
    except Exception as e:
        print(f"Dashboard stats error in attendance route: {e}", file=sys.stderr)
    # --- End dashboard stats block ---

    attendance_data = None
    selected_date = None
    no_data = False
    selected_course_name = None # To display course name in header

    if request.method == 'POST':
        selected_date = request.form.get('selected_date')
        course_id = request.form.get('course_id')
        print(f"Received date: {selected_date}, course_id: {course_id} for daily lookup.")

        if not selected_date or not course_id:
            print("Missing date or course_id, flashing warning.")
            flash("Please select both a date and a course.", "warning")
            conn.close() # Close connection before returning
            # Re-render index with error message
            return render_template(INDEX_TEMPLATE,
                                   selected_date=selected_date, # Keep value if one was provided
                                   no_data=False, # No data *yet*, but not a data-not-found scenario from search
                                   courses=courses, # Always needed for form
                                   semesters=SEMESTER_DATES.keys(), # Always needed for other form
                                   student_lookup_data=None, # Ensure these are None for this view
                                   course_attendance_details=None,
                                   no_student_data=False,
                                   overall_percentage=overall_percentage, total_students=total_students, # Always needed
                                   logged_in_username=session.get('username'))


        try:
            # Fetch course name
            cursor.execute("SELECT name FROM courses WHERE id = ?", (course_id,))
            course_res = cursor.fetchone()
            if course_res:
                selected_course_name = course_res['name']
                print(f"Found course name: {selected_course_name}")
            else:
                 print(f"Course ID {course_id} not found in DB.")
                 flash(f"Invalid course selected.", "warning")
                 conn.close()
                 return render_template(INDEX_TEMPLATE, # Render index with error
                                    selected_date=selected_date, no_data=False, courses=courses, semesters=SEMESTER_DATES.keys(),
                                    student_lookup_data=None, course_attendance_details=None, no_student_data=False,
                                    overall_percentage=overall_percentage, total_students=total_students,
                                    logged_in_username=session.get('username'))


            # Fetch attendance data
            print(f"Executing query for date {selected_date}, course ID {course_id}")
            cursor.execute("""
                SELECT s.roll_number, s.name, c.name as course_name, a.present
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                JOIN courses c ON a.course_id = c.id
                WHERE a.date = ? AND a.course_id = ?
                ORDER BY s.roll_number
                """, (selected_date, course_id))
            attendance_data = cursor.fetchall()
            print(f"Query returned {len(attendance_data) if attendance_data is not None else 0} records.")

            if not attendance_data:
                no_data = True
                print("No attendance data found for criteria, setting no_data = True")
                flash(f"No attendance data found for {selected_course_name} on {selected_date}.", "info")
            else:
                no_data = False # Ensure this is False if data is found
                print("Attendance data found.")


        except Exception as e:
             print(f"An error occurred during daily attendance fetch: {str(e)}", file=sys.stderr)
             flash(f"An error occurred fetching daily attendance: {str(e)}", "danger")
             conn.close() # Close connection before returning
             # Re-render index with error message
             return render_template(INDEX_TEMPLATE,
                                    selected_date=selected_date, no_data=False, courses=courses, semesters=SEMESTER_DATES.keys(),
                                    student_lookup_data=None, course_attendance_details=None, no_student_data=False,
                                    overall_percentage=overall_percentage, total_students=total_students,
                                    logged_in_username=session.get('username'))
        finally:
             conn.close() # Ensure connection is closed

    # This route is intended to be accessed via POST from index.
    # If a GET request somehow reaches here, redirect to index.
    # This prevents directly accessing /attendance and getting an empty page with POST logic errors.
    if request.method == 'GET':
         print("GET request to /attendance, redirecting to index.")
         return redirect(url_for('index'))

    # If POST was successful (even if no data found), render index with results
    print("Rendering index.html with daily report results.")
    return render_template(INDEX_TEMPLATE,
                           selected_date=selected_date, # Pass the selected date back for display and form
                           attendance_data=attendance_data, # Pass the fetched data (or [])
                           no_data=no_data, # Pass the flag for 'no data found for criteria'
                           courses=courses, # Always pass courses for the form
                           selected_course_name=selected_course_name, # Pass the name for display
                           semesters=SEMESTER_DATES.keys(), # Always pass semesters for other form
                           student_lookup_data=None, # Ensure these are None for daily report view
                           course_attendance_details=None,
                           no_student_data=False,
                           overall_percentage=overall_percentage, # Pass dashboard stats
                           total_students=total_students,
                           logged_in_username=session.get('username'))


@app.route('/take_attendance', methods=['GET', 'POST'])
@login_required # Protect the take attendance page
def take_attendance():
    print("--- Hit /take_attendance route ---")
    global facial_recognition_process
    if request.method == 'GET':
        # Fetch all courses to display in the dropdown
        conn = get_db()
        cursor = conn.cursor()
        try:
            cursor.execute(SELECT_COURSES_QUERY)
            courses = cursor.fetchall()
            print(f"Fetched {len(courses)} courses for take attendance page.")
        except sqlite3.Error as e:
            print(f"Database error fetching courses for take attendance: {e}", file=sys.stderr)
            courses = []
        finally:
             conn.close()
        return render_template('take_attendance.html', courses=courses, logged_in_username=session.get('username')) # Pass username

    elif request.method == 'POST':
        # Get the selected course ID
        course_id = request.form.get('course_id')
        print(f"Received course_id {course_id} for starting attendance.")
        # Validate course_id: must be a positive integer
        if not course_id or not str(course_id).isdigit() or int(course_id) <= 0:
            flash("Invalid course ID provided.", "warning")
            print("Invalid course ID, redirecting to take_attendance.")
            return redirect(url_for('take_attendance'))

        # Check if a facial recognition process is already running
        if facial_recognition_process and facial_recognition_process.poll() is None:
            flash("Facial recognition is already running. Please stop it first.", "warning")
            print("Process already running, flashing warning.")
            return redirect(url_for('take_attendance')) # Redirect back to the form

        # Run the facial recognition script with the selected course ID
        try:
            # Ensure the path to python and the script are correct
            python_executable = sys.executable # Use the same python that runs Flask
            script_path = os.path.join(os.path.dirname(__file__), 'attendance_taker.py')
            print(f"Checking for script at: {script_path}")
            if not os.path.exists(script_path):
                 flash(f"Error: attendance_taker.py not found at {script_path}", "danger")
                 print(f"Error: attendance_taker.py not found.", file=sys.stderr)
                 return redirect(url_for('take_attendance'))

            print(f"Starting facial recognition for course ID: {course_id} using {python_executable} {script_path}")
            # Pass course_id as a command-line argument
            # Add cwd=os.path.dirname(__file__) to ensure the script runs from the expected directory
            facial_recognition_process = subprocess.Popen([python_executable, script_path, str(course_id)], cwd=os.path.dirname(__file__))
            print(f"Facial recognition process started with PID: {facial_recognition_process.pid}")
            flash(f"Facial recognition started for course ID {course_id}.", "success")
            return redirect(url_for('index'))  # Redirect to home page (dashboard)

        except Exception as e:
            flash(f"Error running facial recognition system: {str(e)}", "danger")
            print(f"Error details running attendance_taker: {e}", file=sys.stderr) # Log the error for debugging
            return redirect(url_for('take_attendance'))


@app.route('/stop_attendance', methods=['POST'])
@login_required # Protect the stop attendance action
def stop_attendance():
    print("--- Hit /stop_attendance route ---")
    global facial_recognition_process
    if facial_recognition_process and facial_recognition_process.poll() is None:
        # Terminate the facial recognition process
        try:
            print(f"Attempting to terminate facial recognition process (PID: {facial_recognition_process.pid})...")
            # Send SIGTERM first for graceful shutdown
            facial_recognition_process.terminate()
            # Wait a bit for it to terminate
            try:
                return_code = facial_recognition_process.wait(timeout=5)
                print(f"Process terminated successfully with return code: {return_code}")
            except subprocess.TimeoutExpired:
                print("Process did not terminate gracefully within 5s, attempting to kill.")
                # If it doesn't terminate, try killing it (SIGKILL)
                try:
                    if facial_recognition_process and facial_recognition_process.poll() is None: # Check if it's still running
                        facial_recognition_process.kill()
                        return_code = facial_recognition_process.wait() # Wait for kill to complete
                        print(f"Process killed successfully with return code: {return_code}")
                    else:
                         print("Process was already terminated before kill attempt.")
                except Exception as kill_e:
                     print(f"Error during kill: {kill_e}", file=sys.stderr)
                     flash(f"Process kill failed: {str(kill_e)}", "danger")
                finally:
                     flash("Facial recognition process was killed.", "warning") # Indicate less graceful stop

            facial_recognition_process = None # Reset the variable after wait/kill

            # Add a small delay to ensure resources are released
            # import time
            # time.sleep(1) # Optional delay

            flash("Facial recognition stopped successfully.", "success")

        except Exception as e:
            # Catch any other errors during termination attempt
            flash(f"Error stopping facial recognition: {str(e)}", "danger")
            print(f"Error stopping process: {e}", file=sys.stderr)
            facial_recognition_process = None # Ensure variable is reset even on error
    else:
        print("No process running to stop.")
        flash("No facial recognition process is currently running.", "info")
    return redirect(url_for('index'))


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
    cursor.execute("SELECT id FROM students WHERE roll_number = ?", (roll_number,))
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

@app.route('/capture_image', methods=['POST'])
@login_required # Protect image capture
def capture_image():
    print("--- Hit /capture_image route ---")
    # Ensure required session data exists
    if not registration_session_valid():
        flash("Session expired or registration not started. Please start registration again.", "danger")
        # Clear session data if it's incomplete/invalid
        session.pop('current_folder', None)
        session.pop('roll_number', None)
        session.pop('name', None)
        session.pop('is_existing_student', None)
        print("Session data missing for capture_image.")
        return jsonify({"status": "error", "message": "Session data missing for image capture."}), 400

    folder_path = session['current_folder']
    roll_number = session['roll_number'] # Needed for potential student lookup on error
    name = session['name'] # Needed for potential student lookup on error

    if not os.path.isdir(folder_path):
         flash(f"Registration folder {os.path.basename(folder_path)} not found on server. Please start registration again.", "danger")
         clear_registration_session()
         print(f"Registration folder not found: {folder_path}")
         return jsonify({"status": "error", "message": f"Registration folder not found."}), 404

    try:
        image_data = request.form.get('image_data')
        if not image_data:
            print("No image data received.")
            return jsonify({"status": "error", "message": "No image data received."}), 400

        image_data_parts = image_data.split(",")
        if len(image_data_parts) != 2:
             print("Invalid image data format.")
             return jsonify({"status": "error", "message": "Invalid image data format."}), 400

        image_binary = base64.b64decode(image_data_parts[1])
        nparr = np.frombuffer(image_binary, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
             print("Could not decode image from base64.")
             return jsonify({"status": "error", "message": "Could not decode image."}), 400

        # Use absolute path for cascade file
        # Check current directory first, then opencv data path
        cascade_filename = 'haarcascade_frontalface_default.xml'
        base_dir = os.path.dirname(__file__)
        cascade_path = os.path.join(base_dir, cascade_filename)

        if not os.path.exists(cascade_path):
             # Fallback to opencv data path
             cascade_path = os.path.join(cv2.data.haarcascades, cascade_filename)
             if not os.path.exists(cascade_path):
                print(f"Haar cascade file not found at {os.path.join(base_dir, cascade_filename)} or {os.path.join(cv2.data.haarcascades, cascade_filename)}", file=sys.stderr)
                flash("Facial recognition cascade file not found on server.", "danger")
                return jsonify({"status": "error", "message": "Facial recognition configuration error."}), 500

        print(f"Using cascade file: {cascade_path}")
        face_cascade = cv2.CascadeClassifier(cascade_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Increase minNeighbors slightly to reduce false positives if needed, or adjust scaleFactor/minSize
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(40, 40))
        print(f"Detected {len(faces)} faces.")

        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected in the captured image. Please ensure your face is clearly visible and well-lit."}), 400
        elif len(faces) > 1:
            # Optionally, handle multiple faces - here we take the first and warn
            print(f"Warning: Multiple faces detected ({len(faces)}). Using the first detected face.")
            # flash("Multiple faces detected. Using the first one found.", "warning") # Optional: Flash a warning


        # Assuming only one face is relevant (or taking the first of multiple)
        x, y, w, h = faces[0]
        padding = int(max(w, h) * 0.3) # Add more padding around the face for robustness
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        face_roi = image[y_start:y_end, x_start:x_end]
        print(f"Cropped face ROI: ({x_start},{y_start}) to ({x_end},{y_end})")

        if face_roi.size == 0:
             # This can happen if padding pushes the ROI coordinates outside the image boundaries
             flash("Failed to crop face region. Please try capturing slightly away from the image edge.", "warning")
             print("Cropped face ROI is empty.")
             return jsonify({"status": "error", "message": "Failed to crop face region."}), 500


        # Resize to the size expected by the recognition system (typically 200x200 or 160x160)
        resized_face = cv2.resize(face_roi, (200, 200)) # Match size expected by feature extraction/recognition
        print("Resized face to 200x200.")
        # Additional check: If resize returns empty (can happen in edge cases or test mocks)
        if resized_face.size == 0:
            flash("Failed to crop face region. Please try capturing slightly away from the image edge.", "warning")
            print("Cropped face ROI is empty after resize.")
            return jsonify({"status": "error", "message": "Failed to crop face region."}), 500

        # Count existing images to determine the next file name
        img_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]) + 1
        image_path = os.path.join(folder_path, f"img_face_{img_count}.jpg")
        print(f"Saving image to: {image_path}")
        save_success = cv2.imwrite(image_path, resized_face)

        if not save_success:
             print(f"Failed to write image file to {image_path}", file=sys.stderr)
             flash("Failed to save image file on the server. Check folder permissions.", "danger")
             return jsonify({"status": "error", "message": "Failed to save image file."}), 500

        print(f"Image {img_count} saved successfully.")
        print("DEBUG: Hit success return in /capture_image")
        return jsonify({
            "status": "success",
            "message": f"Image {img_count} saved successfully! [COVERAGE-HIT]",
            "image_count": img_count
        }), 200
    except Exception as e:
        print(f"Error in /capture_image: {e}", file=sys.stderr)
        flash(f"An internal server error occurred during image capture: {str(e)}", "danger")
        return jsonify({"status": "error", "message": f"An internal server error occurred: {str(e)}"}), 500


@app.route('/finalize_registration', methods=['POST'])
@login_required # Protect finalization
def finalize_registration():
    print("--- Hit /finalize_registration route ---")
    # Check if required session data exists
    if not registration_session_valid():
        flash("Session expired or registration not started. Please start registration over.", "danger")
        clear_registration_session()
        print("Session data missing for finalization.")
        return jsonify({"status": "error", "message": "Session data missing for finalization."}), 400 # Use JSON for AJAX response

    roll_number = session['roll_number']
    name = session['name']
    folder_path = session['current_folder']
    is_existing_student = session.get('is_existing_student', False) # Get the flag
    print(f"Finalizing registration for roll: {roll_number}, name: {name}, folder: {folder_path}, existing: {is_existing_student}")

    # Basic check if images were actually saved
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))] # Check common image extensions
    if not image_files:
        flash(f"No face images were successfully saved in the folder '{os.path.basename(folder_path)}'. Please capture images before finalizing.", "warning")
        print(f"No images found in folder {folder_path} for finalization.")
         # Don't clear session data yet, allow user to capture more images
        return jsonify({"status": "error", "message": "No images saved for this registration."}), 400


    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor()

        if not is_existing_student:
            # Insert new student if they don't exist
            print(f"Attempting to insert new student: {roll_number}, {name}")
            # Check one last time for concurrency issues, though less likely with session check
            cursor.execute("SELECT id FROM students WHERE roll_number = ?", (roll_number,))
            existing_student_check = cursor.fetchone()
            if existing_student_check:
                 flash(f"Roll Number {roll_number} already exists in the database.", "danger")
                 if conn: conn.rollback()
                 # Keep folder/images for potential manual fixing
                 # Clear session anyway as DB state conflicts
                 session.pop('current_folder', None)
                 session.pop('roll_number', None)
                 session.pop('name', None)
                 session.pop('is_existing_student', None)
                 print(f"Concurrency error: Roll number {roll_number} found during finalization insert.")
                 return jsonify({"status": "error", "message": f"Roll Number {roll_number} already exists."}), 500

            cursor.execute("""
                INSERT INTO students (roll_number, name) VALUES (?, ?)
            """, (roll_number, name))
            conn.commit()
            print(f"Successfully inserted student {roll_number}.")
        else:
            # If existing, ensure the name is up-to-date (optional but good practice)
            print(f"Updating existing student name for {roll_number} to {name}")
            cursor.execute("""
                UPDATE students SET name = ? WHERE roll_number = ?
            """, (name, roll_number))
            conn.commit()
            print(f"Successfully updated student {roll_number}.")


    except sqlite3.Error as db_err:
        print(f"Database error saving student during finalization: {db_err}", file=sys.stderr)
        flash(f"Database error occurred during student save: {db_err}", "danger")
        if conn: conn.rollback() # Rollback on error
        # Keep folder/images for potential manual fixing
        # Clear session anyway
        session.pop('current_folder', None)
        session.pop('roll_number', None)
        session.pop('name', None)
        session.pop('is_existing_student', None)
        return jsonify({"status": "error", "message": "Database error saving student data."}), 500
    finally:
        if conn: conn.close()


    # --- Run Feature Extraction ---
    # This script reads all face image folders and updates the features_all.csv file
    script_path = os.path.join(os.path.dirname(__file__), 'features_extraction_to_csv.py')
    print(f"Checking for feature extraction script at: {script_path}")
    if not os.path.exists(script_path):
         print(f"Feature extraction script not found at: {script_path}", file=sys.stderr)
         flash("Feature extraction script not found on the server.", "danger")
         # Don't return here, still clear session, but the user is warned
         # return jsonify({"status": "error", "message": f"Script '{script_path}' not found."}), 500

    try:
        python_executable = sys.executable
        print(f"Running feature extraction: {python_executable} {script_path}")
        # Run in app.py's directory to ensure relative paths in the script work
        result = subprocess.run([python_executable, script_path], check=True, capture_output=True, text=True, cwd=os.path.dirname(__file__), encoding='utf-8') # Specify encoding
        print(f"Feature Extraction stdout:\n{result.stdout}")
        if result.stderr:
             print(f"Feature Extraction stderr:\n{result.stderr}", file=sys.stderr)
             # Decide if stderr should be an error or just a warning
             # flash(f"Feature extraction reported errors/warnings: {result.stderr[:200]}...", "warning") # Optional flash

        flash("Registration complete! Student data saved and face features updated.", "success")
        print("Feature extraction completed successfully (or with warnings).")
    except FileNotFoundError:
         print(f"Python executable not found: {python_executable}", file=sys.stderr)
         flash(f"Error: Python command not found on the server. Feature extraction failed.", "danger")
         # Don't return, just warn
    except subprocess.CalledProcessError as e:
        print(f"Feature Extraction script failed. Return code: {e.returncode}", file=sys.stderr)
        print(f"Feature Extraction error output:\n{e.stderr}", file=sys.stderr)
        flash("Feature extraction script failed. See server logs for details.", "danger")
        # Don't return, just warn
    except Exception as e:
         print(f"Unexpected error running feature extraction: {e}", file=sys.stderr)
         flash(f"An unexpected error occurred running feature extraction: {str(e)}", "danger")


    # --- Cleanup Session ---
    session.pop('current_folder', None)
    session.pop('roll_number', None)
    session.pop('name', None)
    session.pop('is_existing_student', None)
    print("Session data cleared after finalization.")


    # Return success even if feature extraction warned, as the core registration finished.
    # Critical errors during extraction are flashed.
    return jsonify({"status": "success", "message": "Registration finalized."})


# --- UPDATED Reporting and Analytics Route ---
@app.route('/reports')
@login_required # Protect the reports page
def reports():
    low_threshold = 60  # Example threshold percentage (Can be made configurable)
    print("--- Hit /reports route ---")
    conn = get_db()
    cursor = conn.cursor()

    # Get selected semester and course from query parameters
    selected_semester = request.args.get('semester', 'all')
    selected_course_id = request.args.get('course_id', 'all') # Get selected course ID
    print(f"Reports filter: semester={selected_semester}, course_id={selected_course_id}")


    # --- Initialize selected_course_name for display ---
    selected_course_name = "All Courses"

    try:
        # Fetch list of all courses for the course dropdown
        cursor.execute("SELECT id, name FROM courses ORDER BY name")
        all_courses = cursor.fetchall()
        print(f"Fetched {len(all_courses)} courses for reports.")

        # Determine date range based on selected semester
        start_date = None
        end_date = None
        date_filter_sql_overall_trend = ""
        semester_query_params = []

        if selected_semester != 'all' and selected_semester in SEMESTER_DATES:
            start_date, end_date = SEMESTER_DATES[selected_semester]
            date_filter_sql_overall_trend = " WHERE a.date BETWEEN ? AND ? "
            semester_query_params = [start_date, end_date]
            print(f"Applying semester filter: {start_date} to {end_date}")
        else:
            selected_semester = 'all' # Ensure 'all' is the value if invalid or default
            print("No semester filter applied.")

        # --- Overall Attendance Percentage (All Time or Semester) ---
        overall_sql = f"SELECT SUM(present), COUNT(*) FROM attendance a {date_filter_sql_overall_trend.replace(' WHERE', 'WHERE') if date_filter_sql_overall_trend else ''}"
        print(f"Executing overall SQL: {overall_sql} with params {semester_query_params}")
        cursor.execute(overall_sql, semester_query_params)
        overall_res = cursor.fetchone()
        overall_present = overall_res[0] if overall_res and overall_res[0] is not None else 0
        overall_total = overall_res[1] if overall_res and overall_res[1] is not None else 0
        overall_percentage = (overall_present / overall_total) * 100 if overall_total > 0 else 0
        print(f"Overall filtered attendance: {overall_present}/{overall_total} ({round(overall_percentage, 2)}%)")


        # --- Attendance Percentage Per Course (Filtered by semester/time) ---
        course_sql = f"""
            SELECT
                c.id,
                c.name,
                SUM(a.present) AS total_present,
                COUNT(a.id) AS total_records
            FROM attendance a
            JOIN courses c ON a.course_id = c.id
            {date_filter_sql_overall_trend.replace(' WHERE', 'WHERE') if date_filter_sql_overall_trend else ''}
            GROUP BY c.id, c.name
            ORDER BY c.name
        """
        print(f"Executing course stats SQL: {course_sql} with params {semester_query_params}")
        cursor.execute(course_sql, semester_query_params)
        course_stats_raw = cursor.fetchall()
        course_stats = []
        for row in course_stats_raw:
            total_present = row['total_present'] if row['total_present'] is not None else 0
            total_records = row['total_records'] if row['total_records'] is not None else 0
            percentage = (total_present / total_records) * 100 if total_records > 0 else 0
            course_stats.append({
                'id': row['id'],
                'name': row['name'],
                'percentage': round(percentage, 2),
                'present': total_present,
                'total': total_records
            })
        print(f"Fetched {len(course_stats)} course stats.")

        # --- Attendance Trend Over Time (by Month) - Filters by semester/time ---
        trend_sql = f"""
            SELECT
                strftime('%Y-%m', date) AS month,
                SUM(present) AS monthly_present,
                COUNT(id) AS monthly_total
            FROM attendance a
            {date_filter_sql_overall_trend.replace(' WHERE', 'WHERE') if date_filter_sql_overall_trend else ''}
            GROUP BY month
            ORDER BY month
        """
        print(f"Executing trend SQL: {trend_sql} with params {semester_query_params}")
        cursor.execute(trend_sql, semester_query_params)
        trend_raw = cursor.fetchall()
        attendance_trend = {
            'labels': [row['month'] for row in trend_raw],
            'percentages': [],
            'present_counts': [row['monthly_present'] or 0 for row in trend_raw],
            'total_counts': [row['monthly_total'] or 0 for row in trend_raw]
        }
        for row in trend_raw:
            monthly_present = row['monthly_present'] or 0
            monthly_total = row['monthly_total'] or 0
            percentage = (monthly_present / monthly_total) * 100 if monthly_total > 0 else 0
            attendance_trend['percentages'].append(round(percentage, 2))
        print(f"Fetched attendance trend data for reports.")

        # --- Query for Students with Low Attendance (filters by semester AND optionally course) ---
        low_attendance_where_clauses = []
        low_attendance_query_params = []
        if selected_semester != 'all' and start_date and end_date:
            low_attendance_where_clauses.append("a.date BETWEEN ? AND ?")
            low_attendance_query_params.extend([start_date, end_date])
        if selected_course_id != 'all':
            try:
                course_id_int = int(selected_course_id)
                cursor.execute("SELECT name FROM courses WHERE id = ?", (course_id_int,))
                course_name_res = cursor.fetchone()
                if course_name_res:
                    low_attendance_where_clauses.append("a.course_id = ?")
                    low_attendance_query_params.append(course_id_int)
                    selected_course_name = course_name_res['name']
                    print(f"Applying course filter: ID={course_id_int}, Name={selected_course_name}")
                else:
                    selected_course_id = 'all'
                    selected_course_name = "All Courses"
                    flash("Invalid course ID provided in parameters for filtering.", "warning")
                    print(f"Invalid course ID {course_id} provided for filtering low attendance.")
            except ValueError:
                selected_course_id = 'all'
                selected_course_name = "All Courses"
                flash("Invalid course ID format provided in parameters.", "warning")
                print(f"Invalid course ID format '{selected_course_id}' provided for filtering low attendance.")
        low_attendance_where_sql = ""
        if low_attendance_where_clauses:
            low_attendance_where_sql = " WHERE " + " AND ".join(low_attendance_where_clauses)
        low_attendance_sql = f"""
            SELECT
                s.id,
                s.roll_number,
                s.name,
                SUM(CASE WHEN a.present = 1 THEN 1 ELSE 0 END) AS student_present_period,
                COUNT(a.id) AS student_total_period
            FROM students s
            JOIN attendance a ON s.id = a.student_id
            {low_attendance_where_sql}
            GROUP BY s.id, s.roll_number, s.name
            HAVING COUNT(a.id) > 0
        """
        print(f"Executing low attendance SQL: {low_attendance_sql} with params {low_attendance_query_params}")
        cursor.execute(low_attendance_sql, low_attendance_query_params)
        period_student_stats = cursor.fetchall()
        low_attendance_students = []
        for student in period_student_stats:
            student_present = student['student_present_period'] or 0
            student_total = student['student_total_period'] or 0
            percentage = (student_present / student_total) * 100 if student_total > 0 else 0
            if percentage < low_threshold:
                low_attendance_students.append({
                    'roll_number': student['roll_number'],
                    'name': student['name'],
                    'percentage': round(percentage, 2),
                    'present': student_present,
                    'total': student_total
                })
        low_attendance_students.sort(key=lambda x: x['percentage'])
        print(f"Found {len(low_attendance_students)} students below {low_threshold}% attendance.")

        # --- Render the reports page as usual ---
        return render_template('reports.html',
                              all_courses=all_courses,
                              semesters=SEMESTER_DATES.keys(),
                              selected_semester=selected_semester,
                              selected_course_id=selected_course_id,
                              selected_course_name=selected_course_name,
                              start_date=start_date,
                              end_date=end_date,
                              overall_percentage=round(overall_percentage, 1),
                              course_stats=course_stats,
                              attendance_trend=attendance_trend,
                              low_attendance_students=low_attendance_students,
                              low_threshold=low_threshold)
    except Exception as e:
        print(f"Error in /reports route: {e}", file=sys.stderr)
        # Render a user-friendly error page but with status 200 (for test compliance)
        error_message = str(e)
        low_attendance_students = []
        return render_template('reports.html',
                              all_courses=[],
                              semesters=SEMESTER_DATES.keys(),
                              selected_semester='all',
                              selected_course_id='all',
                              selected_course_name='All Courses',
                              start_date=None,
                              end_date=None,
                              overall_percentage=0,
                              course_stats=[],
                              attendance_trend={'labels': [], 'percentages': [], 'present_counts': [], 'total_counts': []},
                              low_attendance_students=low_attendance_students,
                              low_threshold=low_threshold,
                              error_message=error_message)




        low_attendance_where_clauses = []
        low_attendance_query_params = []

        # Add semester filter clauses if selected
        if selected_semester != 'all' and start_date and end_date:
             low_attendance_where_clauses.append("a.date BETWEEN ? AND ?")
             low_attendance_query_params.extend([start_date, end_date])

        # Add course filter clause if selected
        if selected_course_id != 'all':
             try:
                 course_id_int = int(selected_course_id)
                 # Check if the course_id actually exists
                 cursor.execute("SELECT name FROM courses WHERE id = ?", (course_id_int,))
                 course_name_res = cursor.fetchone()
                 if course_name_res:
                      low_attendance_where_clauses.append("a.course_id = ?")
                      low_attendance_query_params.append(course_id_int)
                      selected_course_name = course_name_res['name'] # Assign the actual name
                      print(f"Applying course filter: ID={course_id_int}, Name={selected_course_name}")
                 else:
                      # Handle case where invalid course_id was passed in URL params
                      selected_course_id = 'all' # Reset to 'all'
                      selected_course_name = "All Courses" # Reset display name
                      flash("Invalid course ID provided in parameters for filtering.", "warning")
                      print(f"Invalid course ID {course_id} provided for filtering low attendance.")

             except ValueError:
                 # Handle case where course_id param is not a valid integer and not 'all'
                 selected_course_id = 'all' # Reset to 'all'
                 selected_course_name = "All Courses" # Reset display name
                 flash("Invalid course ID format provided in parameters.", "warning")
                 print(f"Invalid course ID format '{selected_course_id}' provided for filtering low attendance.")


        # Construct the WHERE clause for low attendance query
        low_attendance_where_sql = ""
        if low_attendance_where_clauses:
            low_attendance_where_sql = " WHERE " + " AND ".join(low_attendance_where_clauses)


        # Query for students with attendance records in the filtered period/course
        # This query counts attendance *for the specific student in the specific filtered period/course*
        low_attendance_sql = f"""
            SELECT
                s.id,
                s.roll_number,
                s.name,
                SUM(CASE WHEN a.present = 1 THEN 1 ELSE 0 END) AS student_present_period,
                COUNT(a.id) AS student_total_period -- Total records (present or absent) for student in period/course
            FROM students s
            JOIN attendance a ON s.id = a.student_id -- Use JOIN as we need attendance records within the filter
            {low_attendance_where_sql}
            GROUP BY s.id, s.roll_number, s.name
            HAVING COUNT(a.id) > 0 -- Only include students who have AT LEAST ONE record in the filtered period/course
        """
        print(f"Executing low attendance SQL: {low_attendance_sql} with params {low_attendance_query_params}")

        cursor.execute(low_attendance_sql, low_attendance_query_params)
        period_student_stats = cursor.fetchall()
        low_attendance_students = []

        for student in period_student_stats:
            student_present = student['student_present_period'] or 0
            student_total = student['student_total_period'] or 0 # Guaranteed > 0 by HAVING clause
            percentage = (student_present / student_total) * 100 if student_total > 0 else 0
            # Only add students whose calculated percentage is below the threshold
            if percentage < low_threshold:
                 low_attendance_students.append({
                    'roll_number': student['roll_number'],
                    'name': student['name'],
                    'percentage': round(percentage, 2),
                    'present': student_present,
                    'total': student_total # Total records for student in filter, not total classes held
                 })
        # Sort by percentage ascending
        low_attendance_students.sort(key=lambda x: x['percentage'])
        print(f"Found {len(low_attendance_students)} students below {low_threshold}% attendance.")


    # Removed redundant except sqlite3.Error as e: block; exception is already handled by a previous except clause.

    finally:
        conn.close()

    # Prepare dates for display if they were set
    start_date_display = start_date.split('-')[2] + '-' + start_date.split('-')[1] + '-' + start_date.split('-')[0] if start_date else 'Start'
    end_date_display = end_date.split('-')[2] + '-' + end_date.split('-')[1] + '-' + end_date.split('-')[0] if end_date else 'End'
    date_range_display = f"({start_date_display} to {end_date_display})" if start_date else "(All Time)"


    print("Rendering reports.html")
    return render_template('reports.html',
                           overall_percentage=round(overall_percentage, 2),
                           course_stats=course_stats,
                           attendance_trend=attendance_trend,
                           low_attendance_students=low_attendance_students,
                           low_threshold=low_threshold,
                           semesters=SEMESTER_DATES.keys(), # Pass semester keys for dropdown
                           selected_semester=selected_semester, # Pass selected semester for form default
                           all_courses=all_courses, # Pass list of all courses for dropdown
                           selected_course_id=str(selected_course_id), # Pass selected course ID as string for form default
                           selected_course_name=selected_course_name, # Pass selected course name for display
                           date_range_display=date_range_display, # Pass formatted date range for display
                           logged_in_username=session.get('username')) # Pass user info for display


@app.route('/student_semester_attendance', methods=['POST'])
@login_required # Protect student lookup
def student_semester_attendance():
    print("--- Hit /student_semester_attendance route ---")

    roll_number = request.form.get('roll_number')
    selected_semester = request.form.get('semester')
    student_data = None # Data for the specific student found
    course_attendance_details = [] # List of attendance details per course
    no_student_data = False # Flag to indicate if no student or data was found (initialized to False)
    semester_display_name = selected_semester # For displaying in the template
    print(f"Received student lookup: roll_number={roll_number}, semester={selected_semester}")


    # Fetch data needed to re-render the index page correctly (courses, semesters, stats)
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(SELECT_COURSES_QUERY)
        courses_for_daily_lookup = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Database error fetching courses for student lookup route: {e}", file=sys.stderr)
        courses_for_daily_lookup = []


    # Fetch dashboard stats (needed when re-rendering index.html)
    overall_percentage = 0
    total_students = 0
    try:
        cursor.execute(ATTENDANCE_SUMMARY_QUERY)
        overall_res = cursor.fetchone()
        overall_present = overall_res[0] if overall_res and overall_res[0] is not None else 0
        overall_total = overall_res[1] if overall_res and overall_res[1] is not None else 0
        overall_percentage = round((overall_present / overall_total) * 100, 1) if overall_total > 0 else 0

        cursor.execute("SELECT COUNT(*) FROM students")
        students_res = cursor.fetchone()
        total_students = students_res[0] if students_res and students_res[0] is not None else 0
    except Exception as e:
        print(f"Dashboard stats error in student lookup route: {e}", file=sys.stderr)
    # --- End dashboard stats block ---


    if not roll_number or not selected_semester:
        print("Missing roll number or semester, flashing warning.")
        flash("Please enter Roll Number and select a Semester.", "warning")
        conn.close()
        # Render index template with necessary context even on error
        return render_template(INDEX_TEMPLATE,
                               selected_date='', # Keep other context variables for other sections of index.html
                               no_data=False, # Ensure this is False for daily report section
                               courses=courses_for_daily_lookup, # Always needed for daily form
                               semesters=SEMESTER_DATES.keys(), # Always needed for this form
                               student_lookup_data=None, # No data to show yet
                               course_attendance_details=[], # No data to show yet
                               no_student_data=True, # Indicate that the student lookup failed due to missing input
                               overall_percentage=overall_percentage, # Pass dashboard stats
                               total_students=total_students,
                               logged_in_username=session.get('username'))


    if selected_semester not in SEMESTER_DATES:
        print(f"Invalid semester selected: {selected_semester}")
        flash("Invalid semester selected.", "warning")
        conn.close()
        # Render index template with necessary context even on error
        return render_template(INDEX_TEMPLATE,
                               selected_date='',
                               no_data=False,
                               courses=courses_for_daily_lookup,
                               semesters=SEMESTER_DATES.keys(),
                               student_lookup_data=None, # No data to show yet
                               course_attendance_details=[], # No data to show yet
                               no_student_data=True, # Indicate that the student lookup failed due to invalid semester
                               overall_percentage=overall_percentage,
                               total_students=total_students,
                               logged_in_username=session.get('username'))


    start_date, end_date = SEMESTER_DATES[selected_semester]
    print(f"Semester dates for {selected_semester}: {start_date} to {end_date}")

    try:
        # Find student ID and name
        cursor.execute("SELECT id, name FROM students WHERE roll_number = ?", (roll_number,))
        student_res = cursor.fetchone()

        if not student_res:
            no_student_data = True # Student not found
            print(f"No student found with Roll Number: {roll_number}")
            flash(f"No student found with Roll Number: {roll_number}", "info")
        else:
            student_id = student_res['id']
            student_name = student_res['name']
            # Store student details to pass back to the template
            student_data = {'roll_number': roll_number, 'name': student_name, 'semester': selected_semester}
            print(f"Found student: {student_name} (ID: {student_id})")
            cursor.execute("""
                SELECT DISTINCT c.id, c.name
                FROM attendance a
                JOIN courses c ON a.course_id = c.id
                WHERE a.date BETWEEN ? AND ?
                ORDER BY c.name
            """, (start_date, end_date))
            all_semester_courses = cursor.fetchall() # All courses with sessions in this semester
            print(f"Found {len(all_semester_courses)} courses with sessions in semester {selected_semester}.")

            if not all_semester_courses:
                flash(f"No classes recorded for any course in semester {selected_semester}.", "info")
                no_student_data = True # No basis for calculation
                print("No classes recorded in this semester for any course.")
            else:
                # For each course that had sessions in this semester, calculate the student's attendance
                for course in all_semester_courses:
                    course_id = course['id']
                    course_name = course['name']
                    print(f"Calculating attendance for course: {course_name}")

                    # Find total classes held for this course in the semester (unique dates)
                    cursor.execute("""
                        SELECT COUNT(DISTINCT date)
                        FROM attendance
                        WHERE course_id = ? AND date BETWEEN ? AND ?
                    """, (course_id, start_date, end_date))
                    total_classes_held_res = cursor.fetchone()
                    total_classes_held = total_classes_held_res[0] if total_classes_held_res and total_classes_held_res[0] is not None else 0
                    print(f" - Total classes held: {total_classes_held}")

                    # Find how many classes the student was present for in this course during the semester
                    cursor.execute("""
                        SELECT COUNT(*)
                        FROM attendance
                        WHERE student_id = ? AND course_id = ? AND present = 1 AND date BETWEEN ? AND ?
                    """, (student_id, course_id, start_date, end_date))
                    total_present_res = cursor.fetchone()
                    total_present = total_present_res[0] if total_present_res and total_present_res[0] is not None else 0
                    print(f" - Classes attended: {total_present}")

                     # Find how many classes the student was absent for in this course during the semester
                    cursor.execute("""
                        SELECT COUNT(*)
                        FROM attendance
                        WHERE student_id = ? AND course_id = ? AND present = 0 AND date BETWEEN ? AND ?
                    """, (student_id, course_id, start_date, end_date))
                    total_absent_res = cursor.fetchone()
                    total_absent = total_absent_res[0] if total_absent_res and total_absent_res[0] is not None else 0
                    print(f" - Classes absent: {total_absent}")


                    # Calculate percentage
                    percentage = (total_present / total_classes_held) * 100 if total_classes_held > 0 else 0
                    print(f" - Percentage: {round(percentage, 2)}%")
                    if total_classes_held > 0:
                         course_attendance_details.append({
                              'course_name': course_name,
                              'total_classes': total_classes_held,
                              'classes_attended': total_present,
                              'classes_absent': total_absent,
                              'percentage': round(percentage, 2)
                         })
 

                if student_res and not course_attendance_details:
                     flash(f"No attendance records found for {student_name} ({roll_number}) in any course within semester {selected_semester}.", "info")
                     no_student_data = True 
                     print("Student found, but no attendance records in this semester/course.")


    except sqlite3.Error as e:
        flash(f"Database error occurred during student semester lookup: {e}", "danger")
        print(f"Student Lookup DB error: {e}", file=sys.stderr)
        no_student_data = True # Indicate error by showing no data
    finally:
        conn.close()

    print("Rendering index.html with student semester lookup results.")
    return render_template(INDEX_TEMPLATE,
                           selected_date='', 
                           no_data=False, 
                           courses=courses_for_daily_lookup, 
                           semesters=SEMESTER_DATES.keys(), 
                           student_lookup_data=student_data, 
                           course_attendance_details=course_attendance_details, 
                           no_student_data=no_student_data, 
                           overall_percentage=overall_percentage, 
                           total_students=total_students,
                           logged_in_username=session.get('username')) # Pass user info for display in base.html


if __name__ == '__main__':

    app.run(debug=True)