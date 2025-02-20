import cv2
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\runs\train\racket\weights\best.pt') 

# Load the video
video_path = r'C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\data\WhatsApp Video 2024-11-04 at 22.32.53_86927ab8.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_path = r'C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\data\net3_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection on the current frame
    results = model(frame)  # Detect objects on the frame

    # Draw bounding boxes and labels on the frame
    for *box, conf, cls in results.xyxy[0].cpu().numpy():  # Convert tensors to numpy for drawing
        x1, y1, x2, y2 = map(int, box)  # Get box coordinates
        label = f"{model.names[int(cls)]} {conf:.2f}"  # Get class name and confidence

        # Draw the box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Write the frame with detections to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Video', frame)

    # Press 'q' to stop the video early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture, video writer, and close display window
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_path}")
