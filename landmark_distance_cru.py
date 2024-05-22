import dlib
import cv2
from utils import get_face_chip, normalize_landmarks, read_landmarks, compare_landmarks_procrustes 

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
    face_landmarks = []
    landmarks1 = None

    # Check if any face is detected
    if len(faces) > 0:
        for face in faces:
            landmarks = predictor(img, face)
            face_chip = get_face_chip(img, landmarks)
            new_faces = detector(face_chip, 1)
        
            for new_face in new_faces:
                new_landmarks = predictor(face_chip, new_face)
                
                for n in range(0, 68):
                    x = new_landmarks.part(n).x
                    y = new_landmarks.part(n).y
                    face_landmarks.append((x, y))
            landmarks1 = normalize_landmarks(face_landmarks, face_chip.shape)

        cv2.imshow("Output", face_chip)
        cv2.waitKey(2000)
        break
    else:
        print("얼굴이 검출되지 않았습니다. 다시 시도해주세요")
    if landmarks1 is not None:
        break

landmarks2 = read_landmarks(filename)

result = compare_landmarks_procrustes(landmarks1, landmarks2, 0.005)

print("Landmarks similarity:", result)

# Close all windows
cap.release()
cv2.destroyAllWindows()
