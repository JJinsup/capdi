import dlib
import cv2
from utils import get_face_chip, normalize_landmarks, read_landmarks, compare_landmarks

landmarks_list = []

student_id=input("학번을 입력하세요:") #input안에 회원가입창에서 입력받은 값을 넣어주면될듯
filename=f"{student_id}.txt"

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

# Read the first frame from the video
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
        with open("landmark.txt", 'w') as f:
            for (x, y) in normalized_landmarks:
                f.write(f"{x},{y}\n")

        # loop over the (x, y)-coordinates for the facial landmark
        # Show the frame with facial landmarks
        cv2.imshow("Output", face_chip)
        cv2.waitKey(2000)
        break

landmarks1 = read_landmarks('landmark.txt')
landmarks2 = read_landmarks(filename)

result = compare_landmarks(landmarks1, landmarks2, 0.03)

print("Landmarks similarity:", result)

# Close all windows
cap.release()
cv2.destroyAllWindows()
