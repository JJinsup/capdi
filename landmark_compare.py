from imutils import face_utils
import numpy as np
import dlib
import cv2

def read_landmarks(filename):
    landmarks = []
    with open(filename, 'r') as f:
        for line in f:
            x, y = map(int, line.strip().split(','))
            landmarks.append((x, y))
    return landmarks

def compare_landmarks(landmarks1, landmarks2, threshold=10):
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

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 0)

# Check if any face is detected
if len(rects) > 0:
    shape = predictor(gray, rects[0])
    shape = face_utils.shape_to_np(shape)

    # Save the landmarks to a text file
    with open('landmarkks.txt', 'w') as f:
        for (x, y) in shape:
            f.write(f"{x},{y}\n")

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

# Show the frame with facial landmarks
cv2.imshow("Output", frame)
cv2.waitKey(0)

landmarks1 = read_landmarks('landmarks.txt')
landmarks2 = read_landmarks('landmarks_me.txt')

result = compare_landmarks(landmarks1, landmarks2)

# Close all windows
cv2.destroyAllWindows()
cap.release()

print("Landmarks similarity:", result)
