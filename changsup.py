import cv2
import pyrealsense2 as rs
import torch
import numpy as np
import sys
import random
import tkinter as tk
from tkinter import messagebox
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add YOLOv5 directory to system path
sys.path.append('/path/to/yolov5')  # Replace with the actual path to the yolov5 directory

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device

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

def calculate_area(depth_image, depth_scale, box, max_depth=3.0, angle=-45):
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Ensure the indices are within bounds
    if cx >= depth_image.shape[1] or cy >= depth_image.shape[0]:
        return None

    D = depth_image[cy, cx] * depth_scale

    if D <= 0 or D > max_depth:
        return None

    # 카메라 기울기 고려
    theta_rad = np.radians(angle)
    D_ground = D * np.cos(theta_rad)

    # Horizontal FOV in radians
    fov_h = 87  # degrees
    fov_h_rad = np.radians(fov_h)

    # Calculate width in meters at ground level
    width_in_meters = 2 * D_ground * np.tan(fov_h_rad / 2)

    # Calculate pixels per meter
    pixel_per_meter_h = 640 / width_in_meters

    return cx, cy, pixel_per_meter_h, D

def draw_3d_shapes(cx, cy, pixel_per_meter_h, D, detections):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Circle parameters
    radius = 0.564  # Radius of the circle with area 1 square meter
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = radius * np.cos(theta)
    circle_z = radius * np.sin(theta) + D

    # Plot circle
    ax.plot(circle_x, [0]*len(circle_x), circle_z, 'b', label='Circle (1m²)')

    # Count people within the circle
    people_count = 0
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        person_cx = (x1 + x2) / 2
        person_cy = (y1 + y2) / 2

        distance = np.sqrt(((person_cx - cx) / pixel_per_meter_h)**2 + ((person_cy - cy) / pixel_per_meter_h)**2)
        if distance <= radius:
            people_count += 1

    # Display count on the plot
    ax.text2D(0.05, 0.95, f'People in 1m² area: {people_count}', transform=ax.transAxes)

    # Plot settings
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('3D Visualization of People Count within 1m² Circle')
    ax.legend()

    plt.show()

# Initialize YOLOv5 model
device = select_device('')
model = attempt_load('/path/to/yolov5s.pt', map_location=device)  # Specify the correct path to the YOLOv5 model
stride = int(model.stride.max())  # model stride
img_size = check_img_size(640, s=stride)  # check img_size

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

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

        # Preprocess the image for YOLOv5
        img = cv2.resize(color_image, (img_size, img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=0)  # person class

        # Process detections
        detections = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], color_image.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, color_image, label=label, color=(255, 0, 0), line_thickness=2)
                    detections.append(xyxy)

        # Calculate ground area and draw 3D shapes
        if len(detections) > 0:
            cx, cy, pixel_per_meter_h, D = calculate_area(depth_image, depth_frame.get_units(), detections[0])
            if cx is not None:
                draw_3d_shapes(cx, cy, pixel_per_meter_h, D, detections)

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
