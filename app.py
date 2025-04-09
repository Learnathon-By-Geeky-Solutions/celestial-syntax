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
