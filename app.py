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

facial_recognition_process = None
app = Flask(__name__)
# You MUST set a secret key for session management
app.secret_key = os.urandom(24) # Generates a random secret key

FACE_IMAGES_DIR = "data/data_faces_from_camera/"
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)
DB_NAME = 'attendance.db' # Define DB Name centrally

# --- Semester Date Ranges ---
# Define the start and end dates for each semester
# Ensure the date format is 'YYYY-MM-DD'
SEMESTER_DATES = {
    "1.1": ("2022-03-01", "2022-08-31"),
    "1.2": ("2022-09-01", "2023-03-31"),
    "2.1": ("2023-04-01", "2023-08-31"), # Example dates, adjust as needed
    "2.2": ("2023-09-01", "2024-03-31"), # Example dates, adjust as needed
    "3.1": ("2024-04-01", "2024-07-31"), # Example dates, adjust as needed
    "3.2": ("2024-08-01", "2025-01-31"), # Specific dates requested by user
    # Add more semesters as needed
}

# --- Database Helper ---
def get_db():
    """ Function to get a database connection """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    return conn

# --- Routes --- (Keep existing routes like index, attendance, etc.)

@app.route('/')
def index():
    conn = get_db()
    cursor = conn.cursor()

    # Fetch courses for the daily lookup dropdown
    cursor.execute("SELECT id, name FROM courses")
    courses = cursor.fetchall()

    # --- NEW: Calculate Overall Stats for Dashboard Card ---
    overall_percentage = 0
    total_students = 0
    try:
        # 1. Overall Attendance Percentage (All Time)
        cursor.execute("SELECT SUM(present), COUNT(*) FROM attendance")
        overall_res = cursor.fetchone()
        overall_present = overall_res[0] if overall_res and overall_res[0] is not None else 0
        overall_total = overall_res[1] if overall_res and overall_res[1] is not None else 0
        overall_percentage = round((overall_present / overall_total) * 100, 1) if overall_total > 0 else 0

        # 2. Total Students Registered
        cursor.execute("SELECT COUNT(*) FROM students")
        students_res = cursor.fetchone()
        total_students = students_res[0] if students_res and students_res[0] is not None else 0

    except sqlite3.Error as e:
        print(f"Database error fetching dashboard stats: {e}") # Log error
        # Keep default values (0) if there's an error
    finally:
        conn.close() # Ensure connection is closed

    # Pass all necessary data to the index template
    return render_template('index.html',
                           selected_date='',
                           no_data=False,
                           courses=courses,
                           semesters=SEMESTER_DATES.keys(), # Pass semesters for the dropdown
                           student_lookup_data=None, # Initialize student lookup data
                           course_attendance_details=None, # Initialize course details
                           no_student_data=False,
                           # --- NEW: Pass dashboard stats ---
                           overall_percentage=overall_percentage,
                           total_students=total_students)


@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM courses")
    courses = cursor.fetchall() # Fetch courses for the dropdown
    attendance_data = None
    selected_date = None
    no_data = False
    selected_course_name = None # To display course name in header

    # --- Fetch dashboard stats for Reports & Analytics card ---
    overall_percentage = 0
    total_students = 0
    try:
        cursor.execute("SELECT SUM(present), COUNT(*) FROM attendance")
        overall_res = cursor.fetchone()
        overall_present = overall_res[0] if overall_res and overall_res[0] is not None else 0
        overall_total = overall_res[1] if overall_res and overall_res[1] is not None else 0
        overall_percentage = round((overall_present / overall_total) * 100, 1) if overall_total > 0 else 0

        cursor.execute("SELECT COUNT(*) FROM students")
        students_res = cursor.fetchone()
        total_students = students_res[0] if students_res and students_res[0] is not None else 0
    except Exception as e:
        print(f"Dashboard stats error: {e}")
    # --- End dashboard stats block ---

    if request.method == 'POST':
        selected_date = request.form.get('selected_date')
        course_id = request.form.get('course_id')

        if not selected_date or not course_id:
            flash("Please select both a date and a course.", "warning")
            # Return template with courses even on error
            return render_template('index.html',
                                   selected_date=selected_date,
                                   no_data=False,
                                   courses=courses,
                                   attendance_data=None,
                                   overall_percentage=overall_percentage,
                                   total_students=total_students)

        try:
            # Fetch course name
            cursor.execute("SELECT name FROM courses WHERE id = ?", (course_id,))
            course_res = cursor.fetchone()
            if course_res:
                selected_course_name = course_res['name']

            # Fetch attendance data
            cursor.execute("""
                SELECT s.roll_number, s.name, c.name as course_name, a.present
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                JOIN courses c ON a.course_id = c.id
                WHERE a.date = ? AND a.course_id = ?
                ORDER BY s.roll_number
                """, (selected_date, course_id))
            attendance_data = cursor.fetchall()
            if not attendance_data:
                no_data = True
                flash(f"No attendance data found for {selected_course_name or 'the selected course'} on {selected_date}.", "info")

        except Exception as e:
             flash(f"An error occurred: {str(e)}", "danger")
             # Return template with courses even on error
             return render_template('index.html',
                                    selected_date=selected_date,
                                    no_data=False,
                                    courses=courses,
                                    attendance_data=None,
                                    overall_percentage=overall_percentage,
                                    total_students=total_students)
        finally:
             conn.close()

    return render_template('index.html',
                           selected_date=selected_date,
                           attendance_data=attendance_data,
                           no_data=no_data,
                           courses=courses, # Always pass courses
                           selected_course_name=selected_course_name, # Pass selected course name
                           overall_percentage=overall_percentage,
                           total_students=total_students)





@app.route('/take_attendance', methods=['GET', 'POST'])
def take_attendance():
    global facial_recognition_process
    if request.method == 'GET':
        # Fetch all courses to display in the dropdown
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM courses")
        courses = cursor.fetchall()
        conn.close()
        return render_template('take_attendance.html', courses=courses)

    elif request.method == 'POST':
        # Get the selected course ID
        course_id = request.form.get('course_id')
        if not course_id:
            flash("Please select a valid course.", "warning")
            return redirect(url_for('take_attendance'))

        # Check if a facial recognition process is already running
        if facial_recognition_process and facial_recognition_process.poll() is None:
            flash("Facial recognition is already running. Please stop it first.", "warning")
            return redirect(url_for('take_attendance'))

        # Run the facial recognition script with the selected course ID
        try:
            # Ensure the path to python and the script are correct
            python_executable = sys.executable # Use the same python that runs Flask
            script_path = os.path.join(os.path.dirname(__file__), 'attendance_taker.py')
            if not os.path.exists(script_path):
                 flash(f"Error: attendance_taker.py not found at {script_path}", "danger")
                 return redirect(url_for('take_attendance'))

            print(f"Starting facial recognition for course ID: {course_id} using {python_executable} {script_path}")
            # Pass course_id as a command-line argument
            facial_recognition_process = subprocess.Popen([python_executable, script_path, str(course_id)])
            flash(f"Facial recognition started for course ID {course_id}.", "success")
            return redirect(url_for('index'))  # Redirect to home page

        except Exception as e:
            flash(f"Error running facial recognition system: {str(e)}", "danger")
            print(f"Error details: {e}") # Log the error for debugging
            return redirect(url_for('take_attendance'))


@app.route('/stop_attendance', methods=['POST'])
def stop_attendance():
    global facial_recognition_process
    if facial_recognition_process and facial_recognition_process.poll() is None:
        # Terminate the facial recognition process
        try:
            facial_recognition_process.terminate()
            facial_recognition_process.wait(timeout=5) # Wait a bit for it to terminate
            facial_recognition_process = None
            flash("Facial recognition stopped successfully.", "success")
        except subprocess.TimeoutExpired:
            flash("Facial recognition process did not terminate gracefully, attempting to kill.", "warning")
            if facial_recognition_process: # Check if still exists
                facial_recognition_process.kill()
                facial_recognition_process = None
        except Exception as e:
            flash(f"Error stopping facial recognition: {str(e)}", "danger")
            facial_recognition_process = None # Reset even if error
    else:
        flash("No facial recognition process is currently running.", "info")
    return redirect(url_for('index'))


# --- Face Registration Routes --- (Keep existing logic)
@app.route('/register')
def register():
    session.pop('current_folder', None)
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

        safe_name = re.sub(r'[^\w\-]+', '_', name)
        safe_roll = re.sub(r'[^\w\-]+', '_', roll_number)

        existing_folders = [f for f in os.listdir(FACE_IMAGES_DIR) if os.path.isdir(os.path.join(FACE_IMAGES_DIR, f)) and f.startswith("person_")]
        person_ids = []
        for folder in existing_folders:
             match = re.match(r'person_(\d+)_.*', folder)
             if match:
                  person_ids.append(int(match.group(1)))
        next_person_id = max(person_ids) + 1 if person_ids else 1

        folder_name = f"person_{next_person_id}_roll_{safe_roll}_name_{safe_name}"
        folder_path = os.path.join(FACE_IMAGES_DIR, folder_name)

        if os.path.exists(folder_path):
             print(f"Folder already exists, reusing: {folder_path}")
        else:
             os.makedirs(folder_path, exist_ok=True)
             print(f"Created folder: {folder_path}")

        session['current_folder'] = folder_path
        session['roll_number'] = roll_number
        session['name'] = name

        return jsonify({
            "status": "success",
            "message": f"Folder '{folder_name}' created/selected successfully.",
            "folder_path": folder_path
        })
    except Exception as e:
        print(f"Error in /create_folder: {e}")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

@app.route('/capture_image', methods=['POST'])
def capture_image():
    if 'current_folder' not in session:
        return jsonify({"status": "error", "message": "Session expired or folder not created. Please create folder again."}), 400
    folder_path = session['current_folder']
    if not os.path.isdir(folder_path):
         return jsonify({"status": "error", "message": f"Folder {folder_path} not found on server. Please create folder again."}), 404

    try:
        image_data = request.form.get('image_data')
        if not image_data:
            return jsonify({"status": "error", "message": "No image data received."}), 400

        image_data_parts = image_data.split(",")
        if len(image_data_parts) != 2:
             return jsonify({"status": "error", "message": "Invalid image data format."}), 400

        image_binary = base64.b64decode(image_data_parts[1])
        nparr = np.frombuffer(image_binary, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
             return jsonify({"status": "error", "message": "Could not decode image."}), 400

        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
             return jsonify({"status": "error", "message": "Haar cascade file not found."}), 500
        face_cascade = cv2.CascadeClassifier(cascade_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected in the captured image."}), 400

        x, y, w, h = faces[0]
        padding = int(max(w, h) * 0.2)
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        face_roi = image[y_start:y_end, x_start:x_end]

        if face_roi.size == 0:
             return jsonify({"status": "error", "message": "Failed to crop face ROI."}), 500

        resized_face = cv2.resize(face_roi, (200, 200))

        img_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]) + 1
        image_path = os.path.join(folder_path, f"img_face_{img_count}.jpg")
        save_success = cv2.imwrite(image_path, resized_face)

        if not save_success:
             return jsonify({"status": "error", "message": "Failed to save image file."}), 500

        return jsonify({
            "status": "success",
            "message": f"Image {img_count} saved successfully!",
            "image_count": img_count
        })
    except Exception as e:
        print(f"Error in /capture_image: {e}")
        return jsonify({"status": "error", "message": "An internal server error occurred while saving the image."}), 500

@app.route('/finalize_registration', methods=['POST'])
def finalize_registration():
    if 'roll_number' not in session or 'name' not in session:
        return jsonify({"status": "error", "message": "Session expired or user details not found. Please start over."}), 400
    roll_number = session['roll_number']
    name = session['name']

    try:
        conn = None
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO students (roll_number, name) VALUES (?, ?)
                ON CONFLICT(roll_number) DO UPDATE SET name=excluded.name
            """, (roll_number, name))
            conn.commit()
        except sqlite3.Error as db_err:
            print(f"Database error: {db_err}")
            if conn: conn.rollback() # Rollback on error
            return jsonify({"status": "error", "message": "Database update failed."}), 500
        finally:
            if conn: conn.close()

        script_path = 'features_extraction_to_csv.py'
        if not os.path.exists(script_path):
             return jsonify({"status": "error", "message": f"Script '{script_path}' not found."}), 500

        try:
            python_executable = sys.executable
            result = subprocess.run([python_executable, script_path], check=True, capture_output=True, text=True)
            print(f"Feature Extraction stdout: {result.stdout}")
            if result.stderr:
                 print(f"Feature Extraction stderr: {result.stderr}")
        except FileNotFoundError:
             return jsonify({"status": "error", "message": f"'{python_executable}' command not found. Is Python installed and in PATH?"}), 500
        except subprocess.CalledProcessError as e:
            print(f"Feature Extraction error output: {e.stderr}")
            return jsonify({"status": "error", "message": "Failed to run feature extraction script. Check server logs for details."}), 500

        session.pop('current_folder', None)
        session.pop('roll_number', None)
        session.pop('name', None)

        return jsonify({"status": "success", "message": "Registration complete! Features updated."})
    except Exception as e:
        print(f"Error in /finalize_registration: {e}")
        return jsonify({"status": "error", "message": f"An internal server error occurred: {str(e)}"}), 500


# --- UPDATED Reporting and Analytics Route ---
@app.route('/reports')
def reports():
    conn = get_db()
    cursor = conn.cursor()

    # Get selected semester and course from query parameters
    selected_semester = request.args.get('semester', 'all')
    selected_course_id = request.args.get('course_id', 'all') # Get selected course ID

    # --- FIX: Initialize selected_course_name ---
    # This ensures the variable exists in the local scope even if no specific course is selected
    selected_course_name = None # Or maybe "All Courses" if you prefer that default display

    # Fetch list of all courses for the course dropdown
    cursor.execute("SELECT id, name FROM courses ORDER BY name") # Added ORDER BY
    all_courses = cursor.fetchall() # Fetch all courses


    # Determine date range based on selected semester
    start_date = None
    end_date = None
    date_filter_sql = ""
    semester_query_params = [] # Use a separate variable for semester params

    if selected_semester != 'all' and selected_semester in SEMESTER_DATES:
        start_date, end_date = SEMESTER_DATES[selected_semester]
        date_filter_sql = " WHERE a.date BETWEEN ? AND ? "
        semester_query_params = [start_date, end_date]
        print(f"Filtering reports for semester {selected_semester}: {start_date} to {end_date}") # Debugging
    else:
        selected_semester = 'all' # Ensure 'all' is set if invalid semester is passed
        print("Showing reports for all time.") # Debugging
        # For 'all time', date_filter_sql remains empty and semester_query_params is empty

    try:
        # --- Build queries with optional date filtering (for overall and trend) ---
        # These only filter by semester/time, not course
        overall_sql = f"SELECT SUM(present), COUNT(*) FROM attendance a {date_filter_sql.replace(' WHERE', 'WHERE') if date_filter_sql else ''}"
        cursor.execute(overall_sql, semester_query_params) # Use semester_query_params
        overall_res = cursor.fetchone()
        overall_present = overall_res[0] if overall_res and overall_res[0] is not None else 0
        overall_total = overall_res[1] if overall_res and overall_res[1] is not None else 0
        overall_percentage = (overall_present / overall_total) * 100 if overall_total > 0 else 0

        # Attendance Percentage Per Course (This also only filters by semester/time)
        course_sql = f"""
            SELECT
                c.name,
                SUM(a.present) AS total_present,
                COUNT(a.id) AS total_records
            FROM attendance a
            JOIN courses c ON a.course_id = c.id
            {date_filter_sql.replace(' WHERE', 'WHERE') if date_filter_sql else ''}
            GROUP BY c.name
            ORDER BY c.name
        """
        cursor.execute(course_sql, semester_query_params) # Use semester_query_params
        course_stats_raw = cursor.fetchall()
        course_stats = []
        for row in course_stats_raw:
            total_present = row['total_present'] if row['total_present'] is not None else 0
            total_records = row['total_records'] if row['total_records'] is not None else 0
            percentage = (total_present / total_records) * 100 if total_records > 0 else 0
            course_stats.append({
                'name': row['name'],
                'percentage': round(percentage, 2),
                'present': total_present,
                'total': total_records
            })

        # Attendance Trend Over Time (by Month) - Filters by semester/time
        trend_sql = f"""
            SELECT
                strftime('%Y-%m', date) AS month,
                SUM(present) AS monthly_present,
                COUNT(id) AS monthly_total
            FROM attendance a
            {date_filter_sql.replace(' WHERE', 'WHERE') if date_filter_sql else ''}
            GROUP BY month
            ORDER BY month
        """
        cursor.execute(trend_sql, semester_query_params) # Use semester_query_params
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


        # --- Query for Students with Low Attendance (filters by semester AND optionally course) ---
        low_threshold = 60 # Example threshold percentage

        low_attendance_where_clauses = []
        low_attendance_query_params = []

        # Add semester filter clauses if selected
        if selected_semester != 'all' and selected_semester in SEMESTER_DATES:
             low_attendance_where_clauses.append("a.date BETWEEN ? AND ?")
             low_attendance_query_params.extend([start_date, end_date])

        # Add course filter clause if selected
        if selected_course_id != 'all':
             try:
                 # Validate course_id is an integer if not 'all'
                 course_id_int = int(selected_course_id)
                 low_attendance_where_clauses.append("a.course_id = ?")
                 low_attendance_query_params.append(course_id_int)
                 # Fetch the name of the selected course for display
                 cursor.execute("SELECT name FROM courses WHERE id = ?", (course_id_int,))
                 course_name_res = cursor.fetchone() # Use a different variable name
                 if course_name_res:
                      selected_course_name = course_name_res['name'] # Assign to the initialized variable
                 else:
                      # Handle case where invalid course_id was passed
                      selected_course_id = 'all' # Reset to 'all'
                      selected_course_name = None # Keep it None if not found
                      flash("Invalid course selected.", "warning") # Inform the user
             except ValueError:
                 # Handle case where course_id is not a valid integer and not 'all'
                 selected_course_id = 'all' # Reset to 'all'
                 selected_course_name = None # Keep it None
                 flash("Invalid course ID format.", "warning") # Inform the user


        # Construct the WHERE clause for low attendance query
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
            LEFT JOIN attendance a ON s.id = a.student_id
            {low_attendance_where_sql}
            GROUP BY s.id, s.roll_number, s.name
            HAVING COUNT(a.id) > 0 -- Only include students with attendance records in the filtered period/course
        """
        # print(f"Low attendance SQL: {low_attendance_sql}") # Debugging
        # print(f"Low attendance params: {low_attendance_query_params}") # Debugging

        cursor.execute(low_attendance_sql, low_attendance_query_params) # Use low_attendance_query_params
        period_student_stats = cursor.fetchall()
        low_attendance_students = []

        for student in period_student_stats:
            student_present = student['student_present_period'] or 0
            student_total = student['student_total_period'] or 0 # Should be > 0 due to HAVING clause
            percentage = (student_present / student_total) * 100 if student_total > 0 else 0
            if percentage < low_threshold:
                 low_attendance_students.append({
                    'roll_number': student['roll_number'],
                    'name': student['name'],
                    'percentage': round(percentage, 2),
                    'present': student_present,
                    'total': student_total
                 })
        # Sort by percentage ascending
        low_attendance_students.sort(key=lambda x: x['percentage'])


    except sqlite3.Error as e:
        flash(f"Database error fetching reports: {e}", "danger")
        # Set default values on error
        overall_percentage = 0
        course_stats = []
        attendance_trend = {'labels': [], 'percentages': [], 'present_counts': [], 'total_counts': []}
        low_attendance_students = []
        # Ensure dates are None if error occurs before they are set
        if 'start_date' not in locals(): start_date = None
        if 'end_date' not in locals(): end_date = None
        # selected_course_name is already initialized to None

    finally:
        conn.close()

    return render_template('reports.html',
                           overall_percentage=round(overall_percentage, 2),
                           course_stats=course_stats,
                           attendance_trend=attendance_trend,
                           low_attendance_students=low_attendance_students,
                           low_threshold=low_threshold,
                           semesters=SEMESTER_DATES.keys(), # Pass semester keys for dropdown
                           selected_semester=selected_semester, # Pass selected semester
                           all_courses=all_courses, # Pass list of all courses
                           selected_course_id=selected_course_id, # Pass selected course ID
                           selected_course_name=selected_course_name, # Pass selected course name
                           start_date=start_date, # Pass start date
                           end_date=end_date) # Pass end date

@app.route('/student_semester_attendance', methods=['POST'])
def student_semester_attendance():
    roll_number = request.form.get('roll_number')
    selected_semester = request.form.get('semester')
    student_data = None
    course_attendance_details = []
    no_student_data = False
    semester_display_name = selected_semester # For displaying in the template

    # Get courses for the daily report dropdown (needed if redirecting/rendering index)
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM courses")
    courses_for_daily_lookup = cursor.fetchall() # Use a different variable name

    # --- Add this block to fetch dashboard stats ---
    overall_percentage = 0
    total_students = 0
    try:
        cursor.execute("SELECT SUM(present), COUNT(*) FROM attendance")
        overall_res = cursor.fetchone()
        overall_present = overall_res[0] if overall_res and overall_res[0] is not None else 0
        overall_total = overall_res[1] if overall_res and overall_res[1] is not None else 0
        overall_percentage = round((overall_present / overall_total) * 100, 1) if overall_total > 0 else 0

        cursor.execute("SELECT COUNT(*) FROM students")
        students_res = cursor.fetchone()
        total_students = students_res[0] if students_res and students_res[0] is not None else 0
    except Exception as e:
        print(f"Dashboard stats error: {e}")
    # --- End dashboard stats block ---

    if not roll_number or not selected_semester:
        flash("Please enter Roll Number and select a Semester.", "warning")
        conn.close()
        return render_template('index.html',
                               selected_date='',
                               no_data=False,
                               courses=courses_for_daily_lookup,
                               semesters=SEMESTER_DATES.keys(),
                               student_lookup_data=None,
                               course_attendance_details=[],
                               no_student_data=False,
                               overall_percentage=overall_percentage,
                               total_students=total_students)

    if selected_semester not in SEMESTER_DATES:
        flash("Invalid semester selected.", "warning")
        conn.close()
        return render_template('index.html',
                               selected_date='',
                               no_data=False,
                               courses=courses_for_daily_lookup,
                               semesters=SEMESTER_DATES.keys(),
                               student_lookup_data=None,
                               course_attendance_details=[],
                               no_student_data=False,
                               overall_percentage=overall_percentage,
                               total_students=total_students)

    start_date, end_date = SEMESTER_DATES[selected_semester]

    try:
        # Find student ID and name
        cursor.execute("SELECT id, name FROM students WHERE roll_number = ?", (roll_number,))
        student_res = cursor.fetchone()

        if not student_res:
            no_student_data = True
            flash(f"No student found with Roll Number: {roll_number}", "info")
        else:
            student_id = student_res['id']
            student_name = student_res['name']
            student_data = {'roll_number': roll_number, 'name': student_name, 'semester': semester_display_name}


            # Get all unique courses the student had attendance marked for in the semester
            cursor.execute("""
                SELECT DISTINCT c.id, c.name
                FROM attendance a
                JOIN courses c ON a.course_id = c.id
                WHERE a.student_id = ? AND a.date BETWEEN ? AND ?
            """, (student_id, start_date, end_date))
            student_courses = cursor.fetchall()


            # For each course, calculate attendance percentage
            for course in student_courses:
                course_id = course['id']
                course_name = course['name']


                # Find total classes held for this course in the semester (unique dates)
                cursor.execute("""
                    SELECT COUNT(DISTINCT date)
                    FROM attendance
                    WHERE course_id = ? AND date BETWEEN ? AND ?
                """, (course_id, start_date, end_date))
                total_classes_held_res = cursor.fetchone()
                total_classes_held = total_classes_held_res[0] if total_classes_held_res and total_classes_held_res[0] is not None else 0


                # Find how many classes the student was present for in this course during the semester
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM attendance
                    WHERE student_id = ? AND course_id = ? AND present = 1 AND date BETWEEN ? AND ?
                """, (student_id, course_id, start_date, end_date))
                total_present_res = cursor.fetchone()
                total_present = total_present_res[0] if total_present_res and total_present_res[0] is not None else 0


                # Calculate percentage
                percentage = (total_present / total_classes_held) * 100 if total_classes_held > 0 else 0


                course_attendance_details.append({
                    'course_name': course_name,
                    'total_classes': total_classes_held,
                    'classes_attended': total_present,
                    'percentage': round(percentage, 2)
                })

            if not course_attendance_details and student_res: # Student exists but no attendance in this sem
                 flash(f"No attendance records found for {student_name} ({roll_number}) in semester {semester_display_name}.", "info")
                 no_student_data = True # Treat as no data to show


    except sqlite3.Error as e:
        flash(f"Database error occurred: {e}", "danger")
        no_student_data = True # Indicate error by showing no data
    finally:
        conn.close()

    return render_template('index.html',
                           selected_date='', # Keep other context variables if needed
                           no_data=False,
                           courses=courses_for_daily_lookup,
                           semesters=SEMESTER_DATES.keys(), # Pass semesters for the dropdown
                           student_lookup_data=student_data, # The student's details
                           course_attendance_details=course_attendance_details, # List of course attendance
                           no_student_data=no_student_data, # Flag for display
                           overall_percentage=overall_percentage,
                           total_students=total_students)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
