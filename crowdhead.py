import cv2
import pyrealsense2 as rs
import torch
import numpy as np
import sys
import random
import tkinter as tk
from tkinter import messagebox
import time

# Add YOLOv5 directory to system path
sys.path.append('/path/to/yolov5')  # Replace with the actual path to the yolov5 directory

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device, time_sync

# Custom function to plot one box
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * max(img.shape[0:2]))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def show_warning(message):
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Warning", message)
    root.destroy()

def calculate_ground_area(depth_image, depth_scale, box, max_depth=3.0, angle=-45):
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Ensure the indices are within bounds
    if cx >= depth_image.shape[1] or cy >= depth_image.shape[0]:
        return None

    z = depth_image[cy, cx] * depth_scale

    if z <= 0 or z > max_depth:
        return None

    # 카메라 기울기 고려
    theta_rad = np.radians(angle)
    z_ground = z * np.cos(theta_rad)
    z_horizontal = z * np.sin(theta_rad)

    # Horizontal and Vertical FOV in radians
    fov_h = 87  # degrees
    fov_v = 58  # degrees
    fov_h_rad = np.radians(fov_h)
    fov_v_rad = np.radians(fov_v)

    # Calculate width and height in meters at ground level
    width_in_meters = 2 * z_ground * np.tan(fov_h_rad / 2)
    height_in_meters = 2 * z_ground * np.tan(fov_v_rad / 2)

    # Calculate pixels per meter
    pixel_per_meter_h = 640 / width_in_meters
    pixel_per_meter_v = 480 / height_in_meters

    side_length_h = int(1 * pixel_per_meter_h)
    side_length_v = int(1 * pixel_per_meter_v)

    x1_area = max(0, cx - side_length_h // 2)
    y1_area = max(0, cy - side_length_v // 2)
    x2_area = min(640, cx + side_length_h // 2)
    y2_area = min(480, cy + side_length_v // 2)

    return x1_area, y1_area, x2_area, y2_area, pixel_per_meter_h, pixel_per_meter_v, z

def count_heads_in_area(detections, depth_image, depth_scale, max_depth=10.0, area_threshold=1.0):
    head_count = 0
    for det in detections:
        for *xyxy, conf, cls in det:
            result = calculate_ground_area(depth_image, depth_scale, xyxy, max_depth=max_depth)
            if result:
                x1_area, y1_area, x2_area, y2_area, pixel_per_meter_h, pixel_per_meter_v, z = result
                area = ((x2_area - x1_area) / pixel_per_meter_h) * ((y2_area - y1_area) / pixel_per_meter_v)  # Convert to square meters
                if area <= area_threshold:
                    head_count += 1
    return head_count

# YOLOv5 model load
device = select_device('0' if torch.cuda.is_available() else 'cpu')  # Use CUDA if available
model = attempt_load('crowdhuman.pt', map_location=device)  # Load the crowdhuman model
model.to(device).eval()  # Move model to CUDA (if available) and set to evaluation mode
stride = int(model.stride.max())
img_size = check_img_size(640, s=stride)  # Ensure img_size is a multiple of stride

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

# Height parameters
drone_height = 2.0  # meters
threshold = 6  # Warning threshold for number of heads

try:
    while True:
        start_time = time.time()
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Prepare image for YOLOv5
        img = cv2.resize(color_image, (img_size, img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)

        # Perform inference with YOLOv5
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Perform model prediction
        pred = model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, 0.75, 0.45, classes=[1], agnostic=False)  # Only detect heads (class 1)

        # Count detected heads
        detected_heads_count = len(pred[0]) if len(pred) > 0 else 0

        # Only start calculating the ground area if heads are detected
        if detected_heads_count > 0:
            # Process detections and calculate ground areas
            head_count = 0
            ground_area_count = 0
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], color_image.shape).round()

                    for *xyxy, conf, cls in det:
                        label = f'{model.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, color_image, label=label, color=(255, 0, 0), line_thickness=2)

                        # Calculate ground area and draw on image
                        result = calculate_ground_area(depth_image, depth_scale, xyxy)
                        if result:
                            x1_area, y1_area, x2_area, y2_area, pixel_per_meter_h, pixel_per_meter_v, z = result
                            if pixel_per_meter_h != 0 and pixel_per_meter_v != 0:  # Ensure no division by zero
                                cv2.rectangle(color_image, (x1_area, y1_area), (x2_area, y2_area), (0, 0, 255), 2)
                                ground_area_count += 1

            # Count heads in the area of 1m^2 if ground areas can be calculated
            if ground_area_count > 0:
                head_count = count_heads_in_area(pred, depth_image, depth_scale, max_depth=10.0, area_threshold=1.0)
                if head_count >= 6:
                    show_warning("Warning: More than 6 people in 1m^2 area!")
            else:
                head_count = detected_heads_count
                if head_count >= threshold:
                    show_warning(f"Warning: More than {threshold} people detected!")

            cv2.putText(color_image, f'People in 1m^2 area: {head_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('YOLOv5 Detection', color_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

