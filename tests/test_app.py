import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from attendance_taker import Face_Recognizer
import pytest
import sqlite3
import tempfile
from unittest.mock import patch, MagicMock
from app import app, get_db, hash_password, check_password, login_required, SEMESTER_DATES
import bcrypt

# Rest of your test file...

# Fixtures
@pytest.fixture
def face_recognizer():
    return Face_Recognizer(course_id=1)
@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    
    # Use NamedTemporaryFile with delete=False and close it manually
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    app.config['DATABASE'] = tmp.name
    
    try:
        with app.test_client() as client:
            with app.app_context():
                init_db()
            yield client
    finally:
        # Clean up
        os.unlink(tmp.name)
import tempfile
import os

def test_prepopulate_attendance(face_recognizer):
    # Use a persistent temp file for the SQLite database
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, roll_number TEXT)")
        cursor.execute("""CREATE TABLE attendance 
                         (student_id INTEGER, course_id INTEGER, date TEXT, present INTEGER)""")
        # Insert a student
        cursor.execute("INSERT INTO students (roll_number) VALUES ('S001')")
        conn.commit()
        # Patch connect to always return the same connection
        with patch('attendance_taker.sqlite3.connect', return_value=conn):
            face_recognizer.prepopulate_attendance()
        # Reconnect for assertions
        conn2 = sqlite3.connect(db_path)
        cursor2 = conn2.cursor()
        cursor2.execute("SELECT * FROM attendance")
        assert cursor2.fetchone() is not None
        conn2.close()
        conn.close()
    finally:
        os.close(db_fd)
        os.remove(db_path)

def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            course_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            present INTEGER NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id),
            FOREIGN KEY (course_id) REFERENCES courses (id)
        )
    """)
    
    # Insert test data
    hashed_pw = hash_password("testpassword")
    cursor.execute("INSERT INTO teachers (username, password_hash) VALUES (?, ?)", 
                   ("testteacher", hashed_pw))
    
    cursor.execute("INSERT INTO students (roll_number, name) VALUES (?, ?)",
                   ("S001", "Test Student"))
    
    cursor.execute("INSERT INTO courses (name) VALUES (?)", ("Test Course",))
    
    cursor.execute("""
        INSERT INTO attendance (student_id, course_id, date, present) 
        VALUES (1, 1, '2023-01-01', 1)
    """)
    
    conn.commit()
    conn.close()

# Helper functions tests
def test_hash_password():
    password = "sh1100!!"
    hashed = hash_password(password)
    assert isinstance(hashed, str)
    assert len(hashed) > 0
    assert bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def test_check_password():
    password = "sh1100!!"
    hashed = hash_password(password)
    assert check_password(hashed, password) is True
    assert check_password(hashed, "wrongpassword") is False
    assert check_password("invalidhash", password) is False

# Authentication tests
def test_login_success(client):
    response = client.post('/login', data={
        'username': 'admin',
        'password': 'sh1100!!'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b"Logged in successfully" in response.data

def test_login_failure(client):
    response = client.post('/login', data={
        'username': 'wronguser',
        'password': 'wrongpassword'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b"Incorrect username" in response.data

def test_logout(client):
    # Login first
    client.post('/login', data={
        'username': 'admin',
        'password': 'sh1100!!'
    }, follow_redirects=True)
    
    # Then logout
    response = client.post('/logout', follow_redirects=True)
    assert response.status_code == 200
    assert b"You have been logged out" in response.data

# Route tests
def test_protected_routes(client):
    # Try accessing protected route without login
    routes = ['/', '/attendance', '/take_attendance', '/reports', '/register']
    for route in routes:
        response = client.get(route, follow_redirects=True)
        assert response.status_code == 200
        assert b"Please log in" in response.data

def test_index_route(client):
    # Login first
    client.post('/login', data={
        'username': 'admin',
        'password': 'sh1100!!'
    }, follow_redirects=True)
    
    response = client.get('/')
    assert response.status_code == 200
    assert b"Dashboard" in response.data

def test_attendance_report(client):
    # Login first
    client.post('/login', data={
        'username': 'admin',
        'password': 'sh1100!!'
    }, follow_redirects=True)
    
    with client.application.app_context():
        db = get_db()
        # Include all required columns
        db.execute("INSERT INTO courses (name, type) VALUES (?, ?)", 
                  ("Test Course", "lecture"))
        import time
        unique_roll = f"S002_{int(time.time())}"
        db.execute("INSERT INTO students (roll_number, name) VALUES (?, ?)",
                  (unique_roll, "Test Student 2"))
        student_id = db.execute("SELECT id FROM students WHERE roll_number = ?", (unique_roll,)).fetchone()[0]
        db.execute("""INSERT INTO attendance (student_id, course_id, date, present)
                      VALUES (?, 1, '2023-01-01', 1)""", (student_id,))
        db.commit()
    
    response = client.post('/attendance', data={
        'selected_date': '2023-01-01',
        'course_id': '1'
    }, follow_redirects=True)
    assert response.status_code == 200
    print(response.data)
    assert b"Test Student" in response.data

def test_student_lookup(client):
    # Login first
    client.post('/login', data={
        'username': 'admin',
        'password': 'sh1100!!'
    }, follow_redirects=True)
    
    response = client.post('/student_semester_attendance', data={
        'roll_number': 'S001',
        'semester': '1.1'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b"S001" in response.data

# Face registration tests
def test_face_registration_flow(client):
    # Login first
    client.post('/login', data={
        'username': 'admin',
        'password': 'sh1100!!'
    }, follow_redirects=True)

    # Create test directory structure
    os.makedirs("data/data_faces_from_camera", exist_ok=True)
    test_dir = "data/data_faces_from_camera/person_1_roll_S002_name_New_Student"
    os.makedirs(test_dir, exist_ok=True)

    # Test folder creation
    response = client.post('/create_folder', json={
        'name': 'New Student',
        'roll_number': 'S002'
    })
    assert response.status_code == 200

    # Test image capture
    import base64
    # Create a valid base64 image (1x1 black PNG)
    valid_b64_img = (
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAwMB/6Xr7QAAAABJRU5ErkJggg=='
    )
    image_data = f'data:image/png;base64,{valid_b64_img}'
    import numpy as np
    def fake_imdecode(*args, **kwargs):
        return np.zeros((1, 1, 3), dtype=np.uint8)
    with patch('cv2.imdecode', side_effect=fake_imdecode), \
         patch('cv2.imwrite'), \
         patch('cv2.CascadeClassifier') as mock_cascade:
        # Patch detectMultiScale to always return a face
        instance = mock_cascade.return_value
        instance.detectMultiScale.return_value = np.array([[0,0,1,1]])
        response = client.post('/capture_image', data={
            'image_data': image_data
        })
        assert response.status_code == 200

# Error handling tests
def test_invalid_routes(client):
    response = client.get('/nonexistent')
    assert response.status_code == 404

def test_database_error_handling(client):
    # Login first
    client.post('/login', data={
        'username': 'admin',
        'password': 'sh1100!!'
    }, follow_redirects=True)

    # Force a database error
    with patch('sqlite3.connect') as mock_connect:
        mock_connect.side_effect = sqlite3.Error("Test error")
        response = client.get('/')
        assert response.status_code == 500
        assert b"A database error occurred" in response.data