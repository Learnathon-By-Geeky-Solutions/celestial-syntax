import sys
import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime


# Dlib  / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor(r"data/data_dlib/shape_predictor_68_face_landmarks.dat")
face_reco_model = dlib.face_recognition_model_v1(r"data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer:
    def __init__(self, course_id):
        # Initialize course ID
        self.course_id = course_id

        # Initialize face recognition-related attributes
        self.face_roll_number_known_list = []  # For roll numbers
        self.face_name_known_list = []         # For names
        self.face_features_known_list = []     # For features

        # FPS and frame-related attributes
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # Frame count and face tracking attributes
        self.frame_cnt = 0
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0
        self.current_frame_face_X_e_distance_list = []
        self.current_frame_face_position_list = []
        self.current_frame_face_feature_list = []
        self.last_current_frame_centroid_e_distance = 0
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

        # Add this line to initialize the font
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def prepopulate_attendance(self):
        """Insert default 'absent' entries for all students in the course"""
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # Get all students
        cursor.execute("SELECT id FROM students")
        students = cursor.fetchall()
        
        for student in students:
            student_id = student[0]
            # Check if entry exists
            cursor.execute("""
                SELECT * FROM attendance 
                WHERE student_id = ? AND course_id = ? AND date = ?
            """, (student_id, self.course_id, current_date))
            
            if not cursor.fetchone():
                # Insert default 'absent' entry
                cursor.execute("""
                    INSERT INTO attendance (student_id, course_id, date, present)
                    VALUES (?, ?, ?, 0)
                """, (student_id, self.course_id, current_date))
        
        conn.commit()
        conn.close()

    #  "features_all.csv"  / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for row in csv_rd.values:
                roll_number = row[0]  # First column is roll number
                name = row[1]        # Second column is name
                features = row[2:].astype(float).tolist()  # Convert to floats here
                self.face_roll_number_known_list.append(roll_number)
                self.face_name_known_list.append(name)
                self.face_features_known_list.append(features)
            logging.info("Faces in Database: %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        # Convert inputs to numpy arrays of float64
        feature_1 = np.array(feature_1, dtype=np.float64)
        feature_2 = np.array(feature_2, dtype=np.float64)
        dist = np.linalg.norm(feature_1 - feature_2)  # Simplified distance calculation
        return dist

    # / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            #  For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    #  cv2 window / putText on cv2 window
    def draw_note(self, img_rd):
        #  / Add some info on windows
        cv2.putText(img_rd, "Face Recognizer with Deep Learning", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, f"Face_{i + 1}: {self.current_frame_face_name_list[i]}", tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                self.font,
                                0.8, (255, 190, 0),
                                1,
                                cv2.LINE_AA)
    # insert data in database

    def attendance(self, roll_number):
        """Mark student as present for the course and date"""
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # Get student ID from roll number
        cursor.execute("SELECT id FROM students WHERE roll_number = ?", (roll_number,))
        student = cursor.fetchone()
        
        if not student:
            print(f"Student with roll {roll_number} not found!")
            conn.close()
            return
        
        student_id = student[0]
        # Update attendance to present
        cursor.execute("""
            UPDATE attendance 
            SET present = 1 
            WHERE student_id = ? AND course_id = ? AND date = ?
        """, (student_id, self.course_id, current_date))
        
        conn.commit()
        print(f"Marked present: Roll {roll_number}, Course {self.course_id}")
        conn.close()

    #  Face detection and recognition wit OT from input video stream
    def process(self, stream):
        # 1.  Get faces known from "features.all.csv"
        self.prepopulate_attendance()
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                kk = cv2.waitKey(1)

                # 2.  Detect faces for frame X
                faces = detector(img_rd, 0)

                # 3.  Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4.  Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5.  update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1  if cnt not changes
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("scene 1:   No face cnt changes in this frame!!!")

                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1

                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            img_rd = cv2.rectangle(img_rd,
                                                   tuple([d.left(), d.top()]),
                                                   tuple([d.right(), d.bottom()]),
                                                   (255, 255, 255), 2)

                    #  Multi-faces in current frame, use centroid-tracker to track
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_face_cnt):
                        # 6.2 Write names under ROI
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                             self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                             cv2.LINE_AA)
                    self.draw_note(img_rd)

                # 6.2  If cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    logging.debug("scene 2: / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    # 6.2.1  Face cnt decreases: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        logging.debug("  / No faces in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                    # 6.2.2 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        logging.debug("  scene 2.2  Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        # 6.2.2.1 Traversal all the faces in the database
                        for k in range(len(faces)):
                            logging.debug("  For face %d in current frame:", k + 1)
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            self.current_frame_face_X_e_distance_list = []

                            # 6.2.2.2  Positions of faces captured
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 6.2.2.3 
                            # For every faces detected, compare the faces in the database
                            for i in range(len(self.face_features_known_list)):
                                # 
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_known_list[i])
                                    logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    #  person_X
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            # 6.2.2.4 / Find the one with minimum e distance
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))

                            # Inside the process() method, after calculating Euclidean distances:
                            # Inside the process() method, after calculating Euclidean distances:
                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                # Get roll number and name from the list
                                recognized_roll = self.face_roll_number_known_list[similar_person_num]
                                recognized_name = self.face_name_known_list[similar_person_num]
                                logging.debug("Recognized roll: %s", recognized_roll)
                                
                                # Update the name list for display
                                self.current_frame_face_name_list[k] = recognized_name
                                
                                # Mark attendance using roll number
                                self.attendance(recognized_roll)
                            else:
                                logging.debug("Unknown person")

                        # 7.  / Add note on cv2 window
                        self.draw_note(img_rd)

                # 8.  'q'  / Press 'q' to exit
                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

                logging.debug("Frame ends\n\n")
    def run(self):
        # cap = cv2.VideoCapture("video.mp4")  # Get video stream from video file
        cap = cv2.VideoCapture(0)              # Get video stream from camera
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python attendance_taker.py <course_id>")
        sys.exit(1)

    course_id = sys.argv[1]
    recognizer = Face_Recognizer(course_id)  # Pass course_id here
    recognizer.run()

if __name__ == '__main__':
    main()