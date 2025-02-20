import cv2
import torch
import time

# Load YOLOv5 models for racket, shuttle, and net detection
model_racket = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\runs\train\racket\weights\best.pt')
model_shuttle = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\runs\train\shuttle\weights\best.pt')
model_net = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\runs\train\net\weights\best.pt')

# Video file path
video_path = r'C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\data\WhatsApp Video 2024-11-04 at 22.32.53_86927ab8.mp4'
cap = cv2.VideoCapture(video_path)

# Slowdown parameters
fault_slowdown_factor = 0.2  # Adjust playback speed when fault detected (0.2 means 5x slower)
slowdown_duration = 1.5  # Duration (in seconds) to slow down video for each fault

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run each model on the frame
    results_racket = model_racket(frame)
    results_shuttle = model_shuttle(frame)
    results_net = model_net(frame)

    # Extract bounding boxes for each detection
    bboxes_racket = results_racket.xyxy[0].cpu().numpy()  # Move to CPU and convert to numpy for racket
    bboxes_shuttle = results_shuttle.xyxy[0].cpu().numpy()  # Move to CPU and convert to numpy for shuttle
    bboxes_net = results_net.xyxy[0].cpu().numpy()  # Move to CPU and convert to numpy for net

    # Assume net is a single bounding box (simplification)
    net_x1, net_y1, net_x2, net_y2 = bboxes_net[0][:4]

    # Initialize flags for each type of fault
    net_touch_detected = False
    net_block_detected = False
    pre_cross_hit_detected = False

    # 1. Check if racket touches the net
    for racket in bboxes_racket:
        racket_x1, racket_y1, racket_x2, racket_y2 = racket[:4]
        if (racket_x1 < net_x2 and racket_x2 > net_x1 and
            racket_y1 < net_y2 and racket_y2 > net_y1):
            net_touch_detected = True
            cv2.putText(frame, "Net Touch Detected", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (int(racket_x1), int(racket_y1)), (int(racket_x2), int(racket_y2)), (0, 0, 255), 2)

    # 2. Check if shuttle is blocked by the net (Net Block)
    for shuttle in bboxes_shuttle:
        shuttle_x1, shuttle_y1, shuttle_x2, shuttle_y2 = shuttle[:4]
        if shuttle_x2 < net_x1:
            net_block_detected = True
            cv2.putText(frame, "Net Block Detected", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (int(shuttle_x1), int(shuttle_y1)), (int(shuttle_x2), int(shuttle_y2)), (255, 0, 0), 2)

    # 3. Check if shuttle is hit before crossing the net (Pre-Cross Hit)
    for shuttle in bboxes_shuttle:
        shuttle_x1, shuttle_y1, shuttle_x2, shuttle_y2 = shuttle[:4]
        for racket in bboxes_racket:
            racket_x1, racket_y1, racket_x2, racket_y2 = racket[:4]
            if racket_x2 < net_x1 and (racket_y1 < shuttle_y2 and racket_y2 > shuttle_y1):
                pre_cross_hit_detected = True
                cv2.putText(frame, "Pre-Cross Hit Detected", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(racket_x1), int(racket_y1)), (int(racket_x2), int(racket_y2)), (0, 255, 0), 2)
                cv2.rectangle(frame, (int(shuttle_x1), int(shuttle_y1)), (int(shuttle_x2), int(shuttle_y2)), (255, 0, 255), 2)

    # 4. Display fault information if any fault was detected
    if net_touch_detected or net_block_detected or pre_cross_hit_detected:
        cv2.putText(frame, "Fault Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Slow down video when a fault is detected
        start_time = time.time()
        while time.time() - start_time < slowdown_duration:
            # Display the frame multiple times for a slowing effect
            cv2.imshow("Net Fault Detection", frame)
            if cv2.waitKey(int(1000 * fault_slowdown_factor)) & 0xFF == ord('q'):
                break
    else:
        # Display the frame at normal speed
        cv2.imshow("Net Fault Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
