from imutils import face_utils
import numpy as np
import dlib
import cv2
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image

landmarks_list = []

def read_landmarks(filename):
    landmarks = []
    with open(filename, 'r') as f:
        for line in f:
            x, y = map(int, line.strip().split(','))
            landmarks.append((x, y))
    return landmarks

def compare_landmarks(landmarks1, landmarks2, threshold=24):
    if len(landmarks1) != len(landmarks2):
        return False
    distances = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(landmarks1, landmarks2)]
    avg_distance = np.mean(distances)
    return avg_distance <= threshold

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

# Read the first frame from the video
ret, frame = cap.read()

scale = 4
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
height, width = img.shape[:2]
s_height, s_width = height // scale, width // scale
img = cv2.resize(img, (s_width, s_height))

dets = detector(img, 1)

# Check if any face is detected
if len(dets) > 0:
    for i, det in enumerate(dets):
        shape = predictor(img, det)
        left_eye = extract_left_eye_center(shape)
        right_eye = extract_right_eye_center(shape)

        M = get_rotation_matrix(left_eye, right_eye)
        rotated = cv2.warpAffine(img, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

        # cropped 이미지에서 얼굴 랜드마크 추출
        cropped_shape = predictor(rotated, det)
        cropped_shape = face_utils.shape_to_np(cropped_shape)

        landmarks_list.append(cropped_shape)

    # Save the landmarks to a text file
    with open('landmarks_me.txt', 'w') as f:
            for (x, y) in cropped_shape:
                f.write(f"{x},{y}\n")

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in cropped_shape:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

# Show the frame with facial landmarks
cv2.imshow("Output", frame)
cv2.waitKey(0)

landmarks1 = read_landmarks('landmarks_me.txt')
landmarks2 = read_landmarks('landmarks_cropped.txt')

result = compare_landmarks(landmarks1, landmarks2)

# Close all windows
cv2.destroyAllWindows()
cap.release()

print("Landmarks similarity:", result)