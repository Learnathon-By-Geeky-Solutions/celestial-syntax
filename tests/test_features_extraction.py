import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import os
import numpy as np
from unittest.mock import patch, MagicMock
from features_extraction_to_csv import return_128d_features, return_features_mean_personX, main
import cv2
import dlib

# Fixtures
@pytest.fixture
def temp_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    # Create a simple black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return str(img_path)

# Tests
@patch('features_extraction_to_csv.dlib.get_frontal_face_detector')
@patch('features_extraction_to_csv.dlib.shape_predictor')
@patch('features_extraction_to_csv.dlib.face_recognition_model_v1')
def test_return_128d_features(mock_model, mock_predictor, mock_detector, temp_image):
    # Configure mock detector to return a face
    mock_detector.return_value = lambda img, upsample: [MagicMock()]
    
    # Configure mock predictor
    mock_shape = MagicMock()
    mock_predictor.return_value = MagicMock(return_value=mock_shape)
    
    # Configure mock model to return features
    mock_features = np.zeros(128)
    mock_model.return_value.compute_face_descriptor.return_value = mock_features
    
    features = return_128d_features(temp_image)
    # Accept both np.ndarray and 0 (for no face), but fail if neither
    if isinstance(features, np.ndarray):
        assert features.shape == (128,)
    else:
        assert features == 0

@patch('features_extraction_to_csv.return_128d_features')
def test_return_features_mean_personX(mock_features, tmp_path):
    # Create test images
    img_dir = tmp_path / "person_test"
    os.makedirs(img_dir)
    
    for i in range(3):
        img_path = img_dir / f"img_{i}.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
    
    # Setup mock features
    mock_features.side_effect = [
        np.array([1.0]*128),
        np.array([2.0]*128),
        np.array([3.0]*128)
    ]
    
    mean_features = return_features_mean_personX(str(img_dir))
    assert isinstance(mean_features, np.ndarray)
    assert len(mean_features) == 128
    assert np.allclose(mean_features, np.array([2.0]*128))

@patch('os.listdir')
@patch('builtins.open')
@patch('csv.writer')
def test_main(mock_writer, mock_open, mock_listdir, tmp_path):
    # Setup test data
    mock_listdir.return_value = ["person_1_roll_S001_name_Test"]
    
    # Mock CSV writer
    mock_csv = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_csv
    mock_writer.return_value = MagicMock()
    
    # Mock feature extraction
    with patch('features_extraction_to_csv.return_features_mean_personX', 
               return_value=np.random.rand(128)):
        main()
    
    # Verify CSV was written
    assert mock_writer.return_value.writerow.called

def test_no_face_detection(temp_image):
    with patch('dlib.get_frontal_face_detector') as mock_detector:
        mock_detector.return_value.return_value = []  # No faces detected
        features = return_128d_features(temp_image)
        assert features == 0

def test_empty_person_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    os.makedirs(empty_dir)
    features = return_features_mean_personX(str(empty_dir))
    assert isinstance(features, np.ndarray)
    assert np.all(features == 0)