import tkinter as tk
from tkinter import filedialog, Label, Button, ttk
import ttkbootstrap as tb
import cv2
import torch
import os
import shutil
from PIL import Image, ImageTk

# Paths to YOLO models (Modify as needed)
RACKET_MODEL_PATH = r"C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\final models\racket_omen\weights\best.pt"
SHUTTLE_MODEL_PATH = r"C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\final models\shuttle_asustuf\weights\best.pt"  # Change later
NET_MODEL_PATH = r"C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\final models\net\weights\best.pt"  # Change later

# Use torch.hub.load to load the custom YOLOv5 models directly
racket_model = torch.hub.load("ultralytics/yolov5", "custom", path=RACKET_MODEL_PATH)
shuttle_model = torch.hub.load("ultralytics/yolov5", "custom", path=SHUTTLE_MODEL_PATH)
net_model = torch.hub.load("ultralytics/yolov5", "custom", path=NET_MODEL_PATH)

# Function to detect objects in a frame
def detect_objects(model, frame):
    results = model(frame)
    return results.xyxy[0].cpu().numpy()  # Return bounding boxes

# Function to check if racket crosses the net before shuttle
def check_net_cross(racket_pos, shuttle_pos, net_pos):
    if racket_pos is None or racket_pos.size == 0:
        return "No Fault"
    if shuttle_pos is None or shuttle_pos.size == 0:
        return "No Fault"
    if net_pos is None or net_pos.size == 0:
        return "No Fault"

    net_x = net_pos[0]  

    racket_x, shuttle_x = racket_pos[0], shuttle_pos[0]

    if racket_x < net_x and shuttle_x > net_x:
        return "Net Cross Fault"

    return "No Fault"


# Process video and apply detection
def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = "processed_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    fault_detected = "No Fault"

    progress_bar["value"] = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        racket_boxes = detect_objects(racket_model, frame)
        shuttle_boxes = detect_objects(shuttle_model, frame)
        net_boxes = detect_objects(net_model, frame)

        # Get first detected positions
        racket_pos = racket_boxes[0][:2] if len(racket_boxes) > 0 else None
        shuttle_pos = shuttle_boxes[0][:2] if len(shuttle_boxes) > 0 else None
        net_pos = net_boxes[0][:2] if len(net_boxes) > 0 else None

        # Check for fault
        fault_type = check_net_cross(racket_pos, shuttle_pos, net_pos)
        if "Fault" in fault_type:
            fault_detected = fault_type

        # Overlay widget
        frame_with_text = overlay_fault_widget(frame, fault_type)
        out.write(frame_with_text)

        # Update progress bar
        progress_bar["value"] += 100 / frame_count
        root.update_idletasks()

    cap.release()
    out.release()
    return output_path, fault_detected

# Overlay widget on video
def overlay_fault_widget(frame, fault_text):
    overlay = frame.copy()
    h, w, _ = frame.shape
    cv2.rectangle(overlay, (20, h - 80), (w - 20, h - 20), (0, 0, 0), -1)
    color = (0, 0, 255) if "Fault" in fault_text else (0, 255, 0)
    cv2.putText(overlay, fault_text, (50, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return overlay

# GUI Functions
def select_video():
    global input_video
    input_video = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if input_video:
        label_status.config(text=f"Selected: {os.path.basename(input_video)}", foreground="green")
        show_video_thumbnail(input_video)

def process_and_display():
    if not input_video:
        label_status.config(text="Please select a video first!", foreground="red")
        return

    label_status.config(text="Processing video... Please wait.", foreground="blue")
    root.update_idletasks()
    
    output_path, fault_result = process_video(input_video)
    label_status.config(text=f"Fault Detection Result: {fault_result}", foreground="green")
    button_download.config(state="normal")

def download_video():
    save_path = filedialog.asksaveasfilename(defaultextension=".mp4",
                                             filetypes=[("MP4 files", "*.mp4")])
    if save_path:
        shutil.copy("processed_video.mp4", save_path)
        label_status.config(text="Video saved successfully!", foreground="green")

# Display video thumbnail
def show_video_thumbnail(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (200, 150))
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(img)
        label_thumbnail.config(image=img_tk)
        label_thumbnail.image = img_tk

# Initialize Tkinter with ttkbootstrap for better design
root = tb.Window(themename="superhero")
root.title("Badminton Fault Detection")
root.geometry("600x400")

# Title
label_title = tb.Label(root, text="🏸 Badminton Fault Detection 🏸", font=("Arial", 16, "bold"), bootstyle="primary")
label_title.pack(pady=10)

# Thumbnail Preview
label_thumbnail = Label(root)
label_thumbnail.pack(pady=5)

# Status Label
label_status = tb.Label(root, text="Upload a video to analyze faults.", font=("Arial", 12))
label_status.pack(pady=10)

# Progress Bar
progress_bar = ttk.Progressbar(root, length=300, mode="determinate")
progress_bar.pack(pady=10)

# Buttons
button_upload = tb.Button(root, text="📂 Upload Video", command=select_video, bootstyle="primary-outline")
button_upload.pack(pady=5)

button_process = tb.Button(root, text="⚡ Process Video", command=process_and_display, bootstyle="success-outline")
button_process.pack(pady=5)

button_download = tb.Button(root, text="💾 Download Processed Video", command=download_video, bootstyle="danger-outline")
button_download.pack(pady=5)
button_download.config(state="disabled")

# Run GUI
root.mainloop()  