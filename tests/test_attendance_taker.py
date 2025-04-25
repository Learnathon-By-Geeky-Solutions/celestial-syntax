import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import sqlite3
from unittest.mock import patch, MagicMock
from attendance_taker import Face_Recognizer
import numpy as np
import pandas as pd

@pytest.fixture
def face_recognizer():
    return Face_Recognizer(course_id=1)

@pytest.fixture
def mock_db():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, roll_number TEXT, name TEXT)")
    cursor.execute("CREATE TABLE attendance (id INTEGER PRIMARY KEY, student_id INTEGER, course_id INTEGER, date TEXT, present INTEGER)")
    cursor.execute("INSERT INTO students (roll_number, name) VALUES ('S001', 'Test Student')")
    conn.commit()
    yield conn
    conn.close()


def test_get_face_database(tmp_path):
    # Create test CSV
    csv_path = tmp_path / "features_all.csv"
    with open(csv_path, 'w') as f:
        f.write("S001,Test Student," + ",".join(["0.1"]*128) + "\n")
    
    # Create recognizer
    recognizer = Face_Recognizer(course_id=1)
    
    with patch('os.path.exists', return_value=True), \
         patch('pandas.read_csv') as mock_read_csv:
        # Create real DataFrame
        data = [["S001", "Test Student"] + [0.1]*128]
        mock_df = pd.DataFrame(data)
        mock_read_csv.return_value = mock_df
        
        result = recognizer.get_face_database()
        assert result == 1
        assert len(recognizer.face_roll_number_known_list) == 1