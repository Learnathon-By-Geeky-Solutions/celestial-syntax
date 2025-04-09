from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import subprocess  # To run the facial recognition script
from datetime import datetime

app = Flask(__name__)
facial_recognition_process = None


@app.route('/')
def index():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Fetch all courses
    cursor.execute("SELECT id, name FROM courses")
    courses = cursor.fetchall()
    
    conn.close()
    
    return render_template('index.html', selected_date='', no_data=False, courses=courses)


@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM courses")
    courses = cursor.fetchall()

    if request.method == 'POST':
        selected_date = request.form.get('selected_date')
        course_id = request.form.get('course_id')

        # Validate input
        if not selected_date or not course_id:
            flash("Please select both a date and a course.")
            return redirect(url_for('attendance'))

        try:
            # Query attendance data
            cursor.execute("""
                SELECT s.roll_number, s.name, c.name, a.present 
                FROM attendance a 
                JOIN students s ON a.student_id = s.id 
                JOIN courses c ON a.course_id = c.id 
                WHERE a.date = ? AND a.course_id = ?
            """, (selected_date, course_id))
            attendance_data = cursor.fetchall()

            if not attendance_data:
                # No data found - render template with no_data=True
                return render_template('index.html', 
                                       selected_date=selected_date,
                                       no_data=True,
                                       courses=courses)
            else:
                # Data found - render with attendance_data
                return render_template('index.html', 
                                       selected_date=selected_date,
                                       attendance_data=attendance_data,
                                       courses=courses)
        except Exception as e:
            # Handle database errors
            flash(f"An error occurred: {str(e)}")
            return redirect(url_for('attendance'))
        finally:
            conn.close()
    else:
        # Initial GET request - render empty state
        conn.close()
        return render_template('index.html', 
                               selected_date=None,
                               no_data=False,
                               courses=courses)


@app.route('/course_attendance', methods=['GET'])
def course_attendance():
    course_id = request.args.get('course_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    if not course_id or not start_date or not end_date:
        return "Please select a course and a valid date range.", 400
    
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Fetch all students and their attendance for the selected course within the date range
    cursor.execute('''
    SELECT s.roll_number, s.name, COUNT(a.id) AS total_classes, SUM(a.present) AS total_present
    FROM students s
    LEFT JOIN attendance a ON s.id = a.student_id AND a.course_id = ? AND a.date BETWEEN ? AND ?
    GROUP BY s.id
    ''', (course_id, start_date, end_date))
    students_attendance = cursor.fetchall()
    
    # Calculate attendance percentage and mark students with less than 60%
    students_data = []
    for student in students_attendance:
        roll_number, name, total_classes, total_present = student
        total_classes = total_classes or 0
        total_present = total_present or 0
        attendance_percentage = (total_present / total_classes) * 100 if total_classes > 0 else 0
        can_sit_exam = attendance_percentage >= 50
        students_data.append({
            'roll_number': roll_number,
            'name': name,
            'total_classes': total_classes,
            'total_present': total_present,
            'attendance_percentage': round(attendance_percentage, 2),
            'can_sit_exam': can_sit_exam
        })
    
    conn.close()
    return render_template('course_attendance.html', students_data=students_data, start_date=start_date, end_date=end_date)


@app.route('/students_cant_sit_exam', methods=['GET'])
def students_cant_sit_exam():
    course_id = request.args.get('course_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    if not course_id or not start_date or not end_date:
        return "Please select a course and a valid date range.", 400
    
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Fetch course name
    cursor.execute("SELECT name FROM courses WHERE id = ?", (course_id,))
    course_name = cursor.fetchone()
    if not course_name:
        return "Invalid course ID.", 400
    course_name = course_name[0]
    
    # Fetch attendance data for the selected course within the date range
    cursor.execute('''
    SELECT s.roll_number, s.name, COUNT(a.id) AS total_classes, SUM(a.present) AS total_present
    FROM students s
    LEFT JOIN attendance a ON s.id = a.student_id AND a.course_id = ? AND a.date BETWEEN ? AND ?
    GROUP BY s.id
    ''', (course_id, start_date, end_date))
    students_attendance = cursor.fetchall()
    
    # Calculate attendance percentage and filter students with less than 50%
    students_cant_sit_exam = []
    for student in students_attendance:
        roll_number, name, total_classes, total_present = student
        total_classes = total_classes or 0
        total_present = total_present or 0
        attendance_percentage = (total_present / total_classes) * 100 if total_classes > 0 else 0
        if attendance_percentage < 50:
            students_cant_sit_exam.append({
                'roll_number': roll_number,
                'name': name,
                'total_classes': total_classes,
                'total_present': total_present,
                'attendance_percentage': round(attendance_percentage, 2)
            })
    
    conn.close()
    return render_template('students_cant_sit_exam.html', course_name=course_name, students_cant_sit_exam=students_cant_sit_exam, start_date=start_date, end_date=end_date)


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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)