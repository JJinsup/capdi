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

    fov = 87  # horizontal field of view in degrees
    fov_rad = np.radians(fov)
    width_in_meters = 2 * z * np.tan(fov_rad / 2)

    if width_in_meters == 0:
        return None

    pixel_per_meter = 640 / width_in_meters
    side_length = int(pixel_per_meter)

    # Adjust for the camera angle
    offset = int(side_length * np.tan(np.radians(angle)))

    x1_area = max(0, cx - side_length // 2)
    y1_area = max(0, cy - side_length // 2 + offset)
    x2_area = min(640, cx + side_length // 2)
    y2_area = min(480, cy + side_length // 2 + offset)

    return x1_area, y1_area, x2_area, y2_area, pixel_per_meter

def count_people_in_area(detections, depth_image, depth_scale, max_depth=3.0, area_threshold=1.0):
    people_count = 0
    for det in detections:
        for *xyxy, conf, cls in det:
            result = calculate_ground_area(depth_image, depth_scale, xyxy, max_depth=max_depth)
            if result:
                x1_area, y1_area, x2_area, y2_area, pixel_per_meter = result
                area = ((x2_area - x1_area) / pixel_per_meter) * ((y2_area - y1_area) / pixel_per_meter)  # Convert to square meters
                if area <= area_threshold:
                    people_count += 1
    return people_count

# YOLOv5 model load
device = select_device('0' if torch.cuda.is_available() else 'cpu')  # Use CUDA if available
model = attempt_load('yolov5s.pt', map_location=device)  # Replace with the actual path to your yolov5 model
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
threshold = 6  # Warning threshold for number of people

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
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Perform inference with YOLOv5
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Perform model prediction
        pred = model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=[0], agnostic=False)  # Only detect humans (class 0)

        # Count detected people
        detected_people_count = len(pred[0]) if len(pred) > 0 else 0

        # Only start calculating the ground area if humans are detected
        if detected_people_count > 0:
            # Process detections and calculate ground areas
            people_count = 0
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
                            x1_area, y1_area, x2_area, y2_area, pixel_per_meter = result
                            if pixel_per_meter != 0:  # Ensure no division by zero
                                cv2.rectangle(color_image, (x1_area, y1_area), (x2_area, y2_area), (0, 255, 0), 2)
                                ground_area_count += 1

            # Count people in the area of 1m^2 if ground areas can be calculated
            if ground_area_count > 0:
                people_count = count_people_in_area(pred, depth_image, depth_scale, max_depth=3.0, area_threshold=1.0)
                if people_count >= 6:
                    show_warning("Warning: More than 6 people in 1m^2 area!")
            else:
                people_count = detected_people_count
                if people_count >= threshold:
                    show_warning(f"Warning: More than {threshold} people detected!")

            cv2.putText(color_image, f'People in 1m^2 area: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('YOLOv5 Detection', color_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Maintain 10 FPS
        elapsed_time = time.time() - start_time
        sleep_time = max(0, (1.0 / 10) - elapsed_time)
        time.sleep(sleep_time)

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

