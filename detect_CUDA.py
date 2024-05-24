import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from torchvision import transforms
import tkinter as tk
from tkinter import messagebox

# YOLOv5 모델 로드
model_path = 'yolov5s.pt'  # 자신의 YOLOv5 모델 경로로 변경
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)['model'].float()  # CUDA를 사용할 수 있으면 CUDA로 로드
model.to(device).eval()

# 신뢰도 임계값 설정
conf_threshold = 0.2

# RealSense 카메라 초기화
pipeline = rs.pipeline()
config = rs.config()

# 원하는 해상도로 스트림 활성화
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

def show_warning(message):
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Warning", message)
    root.destroy()

def calculate_ground_area(depth_image, depth_scale, box, max_depth=0.5):
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    z = depth_image[cy, cx] * depth_scale

    # 포인트가 유효하고 ground에서 max_depth 이내에 있는지 확인
    if z <= 0 or z > max_depth:
        return None

    # depth z를 기준으로 1미터에 대한 픽셀 길이 계산
    fov = 87  # 수평 시야각 (degree 단위)
    fov_rad = np.radians(fov)
    width_in_meters = 2 * z * np.tan(fov_rad / 2)

    if width_in_meters == 0:
        return None

    pixel_per_meter = 640 / width_in_meters
    side_length = int(pixel_per_meter)

    x1_area = max(0, cx - side_length // 2)
    y1_area = max(0, cy - side_length // 2)
    x2_area = min(640, cx + side_length // 2)
    y2_area = min(480, cy + side_length // 2)

    return x1_area, y1_area, x2_area, y2_area

# 이미지 변환을 위한 transforms 정의
preprocess = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 Tensor로 변환
    transforms.Resize((640, 480)),  # 크기 조정
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 정규화
])

try:
    # 스트리밍 시작
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    camera_working = True
except Exception as e:
    print(f"Depth camera error: {e}")
    camera_working = False

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 이미지를 Tensor로 변환
        input_tensor = preprocess(color_image).unsqueeze(0).to(device)

        # YOLOv5 추론
        with torch.no_grad():
            results = model(input_tensor)

        detections = results.xyxy[0].cpu().numpy()

        ground_people_count = 0
        for *box, conf, cls in detections:
            if cls == 0 and conf >= conf_threshold:  # YOLOv5에서 사람 클래스 (class 0) 및 신뢰도 임계값 확인
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image, f"Person Conf: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                area = calculate_ground_area(depth_image, depth_scale, box)
                if area:
                    x1_area, y1_area, x2_area, y2_area = area
                    cv2.rectangle(color_image, (x1_area, y1_area), (x2_area, y2_area), (255, 0, 0), 2)
                    cv2.putText(color_image, "1m^2 Area", (x1_area, y1_area - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    for *inner_box, inner_conf, inner_cls in detections:
                        if inner_cls == 0 and inner_conf >= conf_threshold:
                            x1_inner, y1_inner, x2_inner, y2_inner = map(int, inner_box)
                            cx_inner = (x1_inner + x2_inner) // 2
                            cy_inner = (y1_inner + y2_inner) // 2
                            if x1_area <= cx_inner <= x2_area and y1_area <= cy_inner <= y2_area:
                                ground_people_count += 1
                else:
                    ground_people_count += 1  # 영역 계산에 실패해도 증가

        cv2.putText(color_image, f"People count in 1m^2: {ground_people_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if ground_people_count >= 2:  # 테스트를 위해 임계값을 2로 설정
            show_warning("Warning: More than 2 people in the 1m^2 area!")

        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

