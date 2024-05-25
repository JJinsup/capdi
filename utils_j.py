import numpy as np
import cv2
import dlib
from scipy.spatial import procrustes
from math import hypot
from datetime import datetime, timedelta

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
    
num_frames = 0
short_cheating_count = 0
long_cheating_count = 0

is_time_counting_eye = False    # 눈동자 이탈 시간 측정용
is_time_counting_head = False   # 고개 이탈 시간 측정용


start_time_eye = None
start_time_head = None
criteria_frame_num = 200
timeduration=timedelta()
cause = 0      # 1 : 눈동자 오른쪽
               # 2 : 눈동자 왼쪽
               # 3 : 고개 오른쪽
               # 4 : 고개 왼쪽
criteria_time = 5       # 짧은 시간 긴 시간 나누는 기준초

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


""" Detect face & eye's location
    and blinking
"""
def get_blinking_ratio(facial_landmarks):
    left_point1 = (facial_landmarks.part(36).x, facial_landmarks.part(36).y)
    right_point1 = (facial_landmarks.part(39).x, facial_landmarks.part(39).y)
    center_top1 = midpoint(facial_landmarks.part(37), facial_landmarks.part(38))
    center_bottom1 = midpoint(facial_landmarks.part(41), facial_landmarks.part(40))

    left_point2 = (facial_landmarks.part(42).x, facial_landmarks.part(42).y)
    right_point2 = (facial_landmarks.part(45).x, facial_landmarks.part(45).y)
    center_top2 = midpoint(facial_landmarks.part(43), facial_landmarks.part(44))
    center_bottom2 = midpoint(facial_landmarks.part(47), facial_landmarks.part(46))

    ver_line_len1 = hypot((center_top1[0] - center_bottom1[0]), (center_top1[1] - center_bottom1[1]))
    hor_line_len1 = hypot((left_point1[0] - right_point1[0]), (left_point1[1] - right_point1[1]))
    ver_line_len2 = hypot((center_top2[0] - center_bottom2[0]), (center_top2[1] - center_bottom2[1]))
    hor_line_len2 = hypot((left_point2[0] - right_point2[0]), (left_point2[1] - right_point2[1]))

    blink_ratio_left = hor_line_len1 / ver_line_len1
    blink_ratio_right = hor_line_len2 / ver_line_len2
    blink_ratio = (blink_ratio_left + blink_ratio_right) / 2

    return blink_ratio



"""Print face's area
"""
def print_face(facial_landmarks, _gray, _frame):
    face_region = np.array([(facial_landmarks.part(0).x, facial_landmarks.part(0).y),
                            (facial_landmarks.part(1).x, facial_landmarks.part(1).y),
                            (facial_landmarks.part(2).x, facial_landmarks.part(2).y),
                            (facial_landmarks.part(3).x, facial_landmarks.part(3).y),
                            (facial_landmarks.part(4).x, facial_landmarks.part(4).y),
                            (facial_landmarks.part(5).x, facial_landmarks.part(5).y),
                            (facial_landmarks.part(6).x, facial_landmarks.part(6).y),
                            (facial_landmarks.part(7).x, facial_landmarks.part(7).y),
                            (facial_landmarks.part(8).x, facial_landmarks.part(8).y),
                            (facial_landmarks.part(9).x, facial_landmarks.part(9).y),
                            (facial_landmarks.part(10).x, facial_landmarks.part(10).y),
                            (facial_landmarks.part(11).x, facial_landmarks.part(11).y),
                            (facial_landmarks.part(12).x, facial_landmarks.part(12).y),
                            (facial_landmarks.part(13).x, facial_landmarks.part(13).y),
                            (facial_landmarks.part(14).x, facial_landmarks.part(14).y),
                            (facial_landmarks.part(15).x, facial_landmarks.part(15).y),
                            (facial_landmarks.part(16).x, facial_landmarks.part(16).y),
                            (facial_landmarks.part(18).x, facial_landmarks.part(18).y),
                            (facial_landmarks.part(23).x, facial_landmarks.part(23).y)], np.int32)

    cv2.polylines(_frame, [face_region], True, (0, 255, 255), 1)



""" Detect eye's gazing
"""
def get_gaze_ratio(eye_points, facial_landmarks, _gray, _frame):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                           (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                           (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                           (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                           (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                           (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    cv2.polylines(_frame, [eye_region], True, (0, 255, 255), 1)

    height, width, _ = _frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 1)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(_gray, _gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    # 눈동자의 흰부분 계산으로 보는 방향 추정
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    #cv2.imshow("left", left_side_threshold)
    #cv2.imshow("right", right_side_threshold)

    # left, right side white 가 10 미만이면 눈 감은것으로 인식
    if left_side_white < 5 or right_side_white < 5:
        _gaze_ratio = 1
    else:
        _gaze_ratio = left_side_white / right_side_white

    return _gaze_ratio



""" Detect head's direction
"""
def get_head_angle_ratio(head_points, facial_landmarks, _frame):
    # 코의 가로선 표시
    nose_region1 = np.array([(facial_landmarks.part(head_points[0]).x, facial_landmarks.part(head_points[0]).y),
                             (facial_landmarks.part(head_points[1]).x, facial_landmarks.part(head_points[1]).y),
                             (facial_landmarks.part(head_points[2]).x, facial_landmarks.part(head_points[2]).y),
                             (facial_landmarks.part(head_points[3]).x, facial_landmarks.part(head_points[3]).y)],
                            np.int32)
    cv2.polylines(_frame, [nose_region1], True, (0, 255, 255), 1)

    # 코의 세로선 표시
    nose_region2 = np.array([(facial_landmarks.part(head_points[4]).x, facial_landmarks.part(head_points[4]).y),
                             (facial_landmarks.part(head_points[5]).x, facial_landmarks.part(head_points[5]).y),
                             (facial_landmarks.part(head_points[6]).x, facial_landmarks.part(head_points[6]).y),
                             (facial_landmarks.part(head_points[7]).x, facial_landmarks.part(head_points[7]).y),
                             (facial_landmarks.part(head_points[8]).x, facial_landmarks.part(head_points[8]).y)],
                            np.int32)
    cv2.polylines(_frame, [nose_region2], True, (0, 255, 255), 1)

    # 코의 왼쪽 기준선 표시
    nose_line_left = np.array([(facial_landmarks.part(head_points[3]).x, facial_landmarks.part(head_points[3]).y),
                               (facial_landmarks.part(head_points[4]).x, facial_landmarks.part(head_points[4]).y)],
                              np.int32)
    cv2.polylines(_frame, [nose_line_left], True, (255, 0, 255), 1)

    # 코의 오른쪽 기준선 표시
    nose_line_right = np.array([(facial_landmarks.part(head_points[3]).x, facial_landmarks.part(head_points[3]).y),
                                (facial_landmarks.part(head_points[8]).x, facial_landmarks.part(head_points[8]).y)],
                               np.int32)
    cv2.polylines(_frame, [nose_line_right], True, (255, 0, 255), 1)

    nose_left_point = (facial_landmarks.part(head_points[4]).x, facial_landmarks.part(head_points[4]).y)
    nose_right_point = (facial_landmarks.part(head_points[8]).x, facial_landmarks.part(head_points[8]).y)
    nose_center_point = (facial_landmarks.part(head_points[3]).x, facial_landmarks.part(head_points[3]).y)

    # 오른쪽 기준선과 왼쪽 기준선 길이 계산
    nose_line_len1 = hypot(nose_left_point[0] - nose_center_point[0], nose_left_point[1] - nose_center_point[1])
    nose_line_len2 = hypot(nose_right_point[0] - nose_center_point[0], nose_right_point[1] - nose_center_point[1])

    if nose_line_len1 > nose_line_len2:
        _head_direction = "right"
        _direction_ratio = nose_line_len1 / nose_line_len2
    else:
        _head_direction = "left"
        _direction_ratio = nose_line_len2 / nose_line_len1

    return _head_direction, _direction_ratio



""" Set criteria
"""
def set_criteria(_head_direction, _head_direction_sum, _criteria_finished, _direction_ratio, _eye_direction_sum):

    global criteria_frame_num, num_frames

    _head_direction_criteria = 0
    _eye_direction_criteria = 0

    num_frames += 1
    if _head_direction == "left" and (not _criteria_finished):
        _head_direction_sum += (_direction_ratio - 1) * (-1)
        # print(head_direction_sum)
    elif _head_direction == "right" and (not _criteria_finished):
        _head_direction_sum += (_direction_ratio - 1)
        # print(head_direction_sum)

    if num_frames == criteria_frame_num:
        _head_direction_criteria = (_head_direction_sum / num_frames)
        print("HEAD : {}".format(_head_direction_criteria))
        _criteria_finished = True

    if num_frames == criteria_frame_num:
        _eye_direction_criteria = (_eye_direction_sum / num_frames)
        print("EYE : {}".format(_eye_direction_criteria))

    return _head_direction_criteria, _eye_direction_criteria, _criteria_finished, num_frames



def warn_eye_direction(_criteria_finished, _gaze_ratio, _eye_direction_criteria, _margin_eye):
    global is_time_counting_eye, start_time_eye, cause, timeduration

    if _criteria_finished:
        if _gaze_ratio < _eye_direction_criteria - _margin_eye:
            if not is_time_counting_eye:
                start_time_eye = datetime.now()
                is_time_counting_eye = True
                print("시간 계산중...")
                cause = 2
            print("눈동자 왼쪽으로 벗어남")

        elif _gaze_ratio > _eye_direction_criteria + _margin_eye:
            if not is_time_counting_eye:
                start_time_eye = datetime.now()
                is_time_counting_eye = True
                print("시간 계산중...")
                cause = 1
            print("눈동자 오른쪽으로 벗어남")

        else:
            if is_time_counting_eye:
                end_time_eye=datetime.now()
                timeduration += datetime.now() - start_time_eye
                is_time_counting_eye = False
                


def warn_head_direction(_criteria_finished, _head_direction_criteria, _head_direction, _margin_head):
    global start_time_head, is_time_counting_head, cause, timeduration

    if _criteria_finished:
        if _head_direction_criteria < 0:  # 왼쪽을 바라볼 때
            if (_head_direction[0] == "left" and _head_direction[1] > 1 - _head_direction_criteria + _margin_head) or \
               (_head_direction[0] == "right" and _head_direction[1] > 1 + _head_direction_criteria + _margin_head):
                if not is_time_counting_head:
                    start_time_head = datetime.now()
                    is_time_counting_head = True
                    print("시간 계산중...")
                    cause = 4 if _head_direction[0] == "left" else 3
                print("고개 {}으로 벗어남".format(_head_direction[0]))

            else:
                if is_time_counting_head:
                    timeduration += datetime.now() - start_time_head
                    is_time_counting_head = False
                    

        elif _head_direction_criteria >= 0:  # 오른쪽을 바라볼 때
            if (_head_direction[0] == "left" and _head_direction[1] > 1 - _head_direction_criteria + _margin_head) or \
               (_head_direction[0] == "right" and _head_direction[1] > 1 + _head_direction_criteria + _margin_head):
                if not is_time_counting_head:
                    start_time_head = datetime.now()
                    is_time_counting_head = True
                    print("시간 계산중...")
                    cause = 4 if _head_direction[0] == "left" else 3
                print("고개 {}으로 벗어남".format(_head_direction[0]))

            else:
                if is_time_counting_head:
                    timeduration += datetime.now() - start_time_head
                    is_time_counting_head = False