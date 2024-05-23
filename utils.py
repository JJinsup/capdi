import numpy as np
import cv2
import dlib
from scipy.spatial import procrustes


LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def extract_eye(shape, eye_indices):
    points = [shape.part(i) for i in eye_indices]
    return list(points)

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]

# 눈 중심을 기준으로 얼굴을 정규화
def get_face_chip(img, landmarks):

    left_eye_center = extract_left_eye_center(landmarks)
    right_eye_center = extract_right_eye_center(landmarks)

    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)

    angle = angle_between_2_points(left_eye_center, right_eye_center)

    M = cv2.getRotationMatrix2D(eye_center, angle, 1)

    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # 새로운 얼굴 크기와 위치로 이동
    face_chip = dlib.get_face_chip(rotated_img, landmarks, size=300, padding=0.3)
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
    if not landmarks:
        return None  # 랜드마크가 비어 있으면 None 반환

    face_center = calculate_face_center(landmarks)
    normalized_landmarks = [( (x - face_center[0]) / img_shape[1], (y - face_center[1]) / img_shape[0]) for x, y in landmarks]
    return normalized_landmarks

#유클리안 거리 계산 알고리즘
def euclidean_distance(p1, p2):

    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

#랜드마크 읽어오기
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
        distances = [euclidean_distance(p1, p2) for p1, p2 in zip(landmarks1, landmarks2)]
        similarity_score = np.mean(distances)
    # Compare similarity score with threshold
        if similarity_score < threshold:
            return similarity_score, True
        else:
            return similarity_score, False
        
def procrustes_analysis(landmarks1, landmarks2):
    """
    Perform Procrustes analysis to compare two sets of landmarks.
    """
    mtx1, mtx2, disparity = procrustes(landmarks1, landmarks2)
    return disparity

def compare_landmarks_procrustes(landmarks1, landmarks2, threshold):
    """
    Compare two sets of landmarks using Procrustes analysis.
    """
    if len(landmarks1) != len(landmarks2):
        raise ValueError("Number of landmarks must be the same for comparison.")
    
    # Normalize the landmarks by scaling them between 0 and 1
    landmarks1 = np.array(landmarks1)
    landmarks2 = np.array(landmarks2)
    
    landmarks1 = (landmarks1 - landmarks1.min(0)) / landmarks1.ptp(0)
    landmarks2 = (landmarks2 - landmarks2.min(0)) / landmarks2.ptp(0)
    
    disparity = procrustes_analysis(landmarks1, landmarks2)
    
    if disparity < threshold:
            return disparity, True
    else:
            return disparity, False