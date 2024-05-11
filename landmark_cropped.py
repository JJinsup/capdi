from imutils import face_utils
import dlib
import cv2
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image

landmarks_list = []

if __name__ == "__main__":
    input_image = "C:/facial-landmarks-recognition/me.jpg"  # 입력 이미지 파일 경로
    scale = 4  # 이미지 축소 비율

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]
    s_height, s_width = height // scale, width // scale
    img = cv2.resize(img, (s_width, s_height))

    dets = detector(img, 1)

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

    # Save landmarks to a text file
    with open('landmarks_cropped.txt', 'w') as f:
            for (x, y) in cropped_shape:
                f.write(f"{x},{y}\n")

    # Close all windows
    cv2.destroyAllWindows()