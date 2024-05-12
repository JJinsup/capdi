from imutils import face_utils
import numpy as np
import dlib
import cv2
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image

landmarks_list = []

def euclidean_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def read_landmarks(file_path):
    """
    Read landmarks from a .txt file.
    """
    landmarks = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split(','))
            landmarks.append((x, y))
    return landmarks

def compare_landmarks(landmarks1, landmarks2, threshold):
    """
    Compare two sets of landmarks using Euclidean distance.
    """
    if len(landmarks1) != len(landmarks2):
        raise ValueError("Number of landmarks must be the same for comparison.")

    distances = []
    for p1, p2 in zip(landmarks1, landmarks2):
        distance = euclidean_distance(p1, p2)
        distances.append(distance)

    # Calculate some measure of overall similarity (e.g., mean distance)
    similarity_score = np.mean(distances)
    
    # Compare similarity score with threshold
    if similarity_score < threshold:
        return similarity_score, True
    else:
        return similarity_score, False

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
    with open('landmarks_I.txt', 'w') as f:
            for (x, y) in cropped_shape:
                f.write(f"{x},{y}\n")

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in cropped_shape:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

# Show the frame with facial landmarks
cv2.imshow("Output", frame)
cv2.waitKey(0)

landmarks1 = read_landmarks('landmarks_I.txt')
landmarks2 = read_landmarks('landmarks_me.txt')

result = compare_landmarks(landmarks1, landmarks2, 4)

# Close all windows
cap.release()
cv2.destroyAllWindows()

print("Landmarks similarity:", result)