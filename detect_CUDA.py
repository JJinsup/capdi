import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import tkinter as tk
from tkinter import messagebox

# Load YOLOv5s model with pretrained weights, use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.eval()

# Set confidence threshold
conf_threshold = 0.2

# Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()

# Enable streams with desired resolution
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

    # Check if the point is valid and within max_depth from the ground
    if z <= 0 or z > max_depth:
        return None

    # Calculate the pixel length for 1 meter based on depth z
    fov = 87  # Horizontal field of view in degrees
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

try:
    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    camera_working = True
except Exception as e:
    print(f"Depth camera error: {e}")
    camera_working = False

# If depth camera is not working, set drone altitude z manually
if not camera_working:
    drone_altitude = 1  # Example altitude
    threshold = drone_altitude + 5

try:
    cap = None
    if not camera_working:
        cap = cv2.VideoCapture(0)
    
    while True:
        if camera_working:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
        else:
            ret, color_image = cap.read()
            if not ret:
                break

        # Image preprocessing using CUDA if available
        color_image = cv2.resize(color_image, (640, 480))  # Ensure the resolution is 640x480
        if torch.cuda.is_available():
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(color_image)
            gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
            channels = gpu_hsv.split()
            channels[2] = cv2.cuda.equalizeHist(channels[2])
            gpu_hsv = cv2.cuda.merge(channels)
            gpu_result = cv2.cuda.cvtColor(gpu_hsv, cv2.COLOR_HSV2BGR)
            color_image = gpu_result.download()
            color_image = cv2.convertScaleAbs(color_image, alpha=1.5, beta=20)  # Adjust contrast and brightness
        else:
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
            color_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            color_image = cv2.convertScaleAbs(color_image, alpha=1.5, beta=20)  # Adjust contrast and brightness

        # YOLOv5 inference
        results = model(color_image)
        detections = results.xyxy[0].cpu().numpy()
        
        ground_people_count = 0
        people_count = 0
        for *box, conf, cls in detections:
            if cls == 0 and conf >= conf_threshold:  # class 0 is person for YOLOv5 and confidence threshold check
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image, f"Person Conf: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if camera_working:
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
                        ground_people_count += 1  # Increment even if area calculation fails
                else:
                    people_count += 1

        if camera_working:
            cv2.putText(color_image, f"People count in 1m^2: {ground_people_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if ground_people_count >= 2:  # Change the threshold to 2 for testing
                show_warning("Warning: More than 2 people in the 1m^2 area!")
        else:
            cv2.putText(color_image, f"People count: {people_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if people_count >= threshold:
                show_warning("Warning: More than threshold people detected!")

        cv2.imshow('RealSense' if camera_working else 'USB Camera', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    if camera_working:
        pipeline.stop()
    elif cap:
        cap.release()
    cv2.destroyAllWindows()
