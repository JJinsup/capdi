from imutils import face_utils
import numpy as np
import dlib
import cv2
from utils import extract_left_eye_center, extract_right_eye_center, angle_between_2_points

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

# 눈 중심을 기준으로 얼굴을 정규화
def get_face_chip(img, landmarks):

    left_eye_center = extract_left_eye_center(landmarks)
    right_eye_center = extract_right_eye_center(landmarks)

    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)

    angle = angle_between_2_points(left_eye_center, right_eye_center)

    M = cv2.getRotationMatrix2D(eye_center, angle, 1)

    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # 새로운 얼굴 크기와 위치로 이동
    face_chip = dlib.get_face_chip(rotated_img, landmarks, size=150, padding=0.25)
    return face_chip

#랜드마크를 사용해서 얼굴 중심 계산
def calculate_face_center(landmarks):

    x_coords = [x for x, y in landmarks]
    y_coords = [y for x, y in landmarks]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return (center_x, center_y)

#랜드마크 정규화
def normalize_landmarks(landmarks, img_shape):

    face_center = calculate_face_center(landmarks)
    normalized_landmarks = [( (x - face_center[0]) / img_shape[1], (y - face_center[1]) / img_shape[0]) for x, y in landmarks]
    return normalized_landmarks


student_id = input("학번을 입력하세요:") # input안에 회원가입창에서 입력받은 값을 넣어주면될듯
filename = f"{student_id}.txt"

landmarks_list = []

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(img, 1)
    landmark_coords = []

    # Check if any face is detected
    if len(faces) > 0:
        for face in faces:
            landmarks = predictor(img, face)
            face_chip = get_face_chip(img, landmarks)
            new_faces = detector(face_chip, 1)
        
            for new_face in new_faces:
                new_landmarks = predictor(face_chip, new_face)
                face_landmarks = []

                for n in range(0, 68):
                    x = new_landmarks.part(n).x
                    y = new_landmarks.part(n).y
                    face_landmarks.append((x, y))
            normalized_landmarks = normalize_landmarks(face_landmarks, face_chip.shape)

        # Save the landmarks to a text file
        with open(filename, 'w') as f:
            for (x, y) in normalized_landmarks:
                f.write(f"{x},{y}\n")

        # Show the frame with facial landmarks
        cv2.imshow("Output", face_chip)
        cv2.waitKey(2000)
        break
    
# Close all windows
cap.release()
cv2.destroyAllWindows()
