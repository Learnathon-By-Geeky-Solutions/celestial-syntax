# Extract features from images and save into "features_all.csv"

import os
import dlib
import csv
import numpy as np
import logging
import cv2

#  Path of cropped faces
path_images_from_camera = "data/data_faces_from_camera/"

#  Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

#  Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

#  Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


#  Return 128D features for single image

def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)
    if img_rd is None:
        return 0
    faces = detector(img_rd, 1)

    logging.info("%-40s %-20s", " Image with faces detected:", path_img)

    # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("no face")
    return face_descriptor


#   Return the mean value of 128D face descriptor for person X

def return_features_mean_personX(path_face_personX):
    features_list = []
    photos_list = os.listdir(path_face_personX)
    
    if not photos_list:
        return np.zeros(128, dtype=np.float32)  # Return zeros if no images
    
    for photo in photos_list:
        img_path = os.path.join(path_face_personX, photo)
        features = return_128d_features(img_path)
        
        if np.any(features):  # Check if features are valid
            features_list.append(features)
        else:
            logging.warning("No face detected in %s", img_path)
    
    if features_list:
        return np.mean(features_list, axis=0)
    else:
        return np.zeros(128, dtype=np.float32)

def main():
    logging.basicConfig(level=logging.INFO)
    path_images_from_camera = "data/data_faces_from_camera/"
    person_list = os.listdir(path_images_from_camera)
    person_list.sort()

    with open("data/features_all.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for person in person_list:
            # Parse folder name
            parts = person.split('_')
            if len(parts) >= 5 and parts[2] == 'roll' and parts[4] == 'name':
                roll = parts[3]
                name = '_'.join(parts[5:])  # Handle names with underscores
            else:
                continue  # Skip invalid formats

            # Get features
            features_mean = return_features_mean_personX(os.path.join(path_images_from_camera, person))
            
            # Create CSV row: [Roll, Name, feature1, feature2, ..., feature128]
            row = [roll, name] + features_mean.tolist()
            writer.writerow(row)
            
            logging.info("Processed: Roll %s, Name %s", roll, name)

if __name__ == '__main__':
    main()