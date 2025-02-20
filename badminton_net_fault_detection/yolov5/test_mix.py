import cv2
import torch
import numpy as np

# Paths to YOLOv5 models (Modify as needed)
RACKET_MODEL_PATH = r"C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\final models\racket_omen\weights\best.pt"
SHUTTLE_MODEL_PATH = r"C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\final models\shuttle_asustuf\weights\best.pt"  # Change later
NET_MODEL_PATH = r"C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\final models\net\weights\best.pt" 

# Load YOLO models
racket_model = torch.hub.load("ultralytics/yolov5", "custom", path=RACKET_MODEL_PATH, force_reload=True)
shuttle_model = torch.hub.load("ultralytics/yolov5", "custom", path=SHUTTLE_MODEL_PATH, force_reload=True)
net_model = torch.hub.load("ultralytics/yolov5", "custom", path=NET_MODEL_PATH, force_reload=True)

# Function to detect objects in a frame
def detect_objects(model, frame):
    results = model(frame)
    return results.xyxy[0].cpu().numpy()  # Return bounding boxes

# Function to check if racket crosses the net before shuttle
def check_net_cross(racket_pos, shuttle_pos, net_pos):
    net_x = net_pos[0]  # Assuming net is vertical at X position

    if racket_pos and shuttle_pos:
        racket_x, shuttle_x = racket_pos[0], shuttle_pos[0]

        if racket_x < net_x and shuttle_x > net_x:  
            return "Net Cross Fault"
    
    return "No Fault"

# Process video
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        racket_boxes = detect_objects(racket_model, frame)
        shuttle_boxes = detect_objects(shuttle_model, frame)
        net_boxes = detect_objects(net_model, frame)

        # Get first detected positions (Modify logic for multiple detections)
        racket_pos = racket_boxes[0][:2] if len(racket_boxes) > 0 else None
        shuttle_pos = shuttle_boxes[0][:2] if len(shuttle_boxes) > 0 else None
        net_pos = net_boxes[0][:2] if len(net_boxes) > 0 else None

        # Check for fault
        fault_type = check_net_cross(racket_pos, shuttle_pos, net_pos)

        # Overlay fault info
        frame_with_text = overlay_fault_widget(frame, fault_type)
        out.write(frame_with_text)

    cap.release()
    out.release()
    print(f"Processed video saved as: {output_path}")

# Overlay widget
def overlay_fault_widget(frame, fault_text):
    overlay = frame.copy()
    h, w, _ = frame.shape
    cv2.rectangle(overlay, (20, h - 80), (w - 20, h - 20), (0, 0, 0), -1)
    color = (0, 0, 255) if "Fault" in fault_text else (0, 255, 0)
    cv2.putText(overlay, fault_text, (50, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return overlay

# Run the system
if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_processed.mp4"
    process_video(input_video, output_video)
