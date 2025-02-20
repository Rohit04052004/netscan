import cv2
import torch

# Load YOLOv5 models for racket, shuttle, and net detection
model_racket = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\runs\train\racket\weights\best.pt')
model_shuttle = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\runs\train\shuttle\weights\best.pt')
model_net = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\runs\train\net\weights\best.pt')

# Path to the image
image_path = r'C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\data\IMG_3304.jpg'
image = cv2.imread(image_path)

# Run each model on the image
results_racket = model_racket(image)
results_shuttle = model_shuttle(image)
results_net = model_net(image)

# Extract bounding boxes for each detection and save cropped images
count = 0
for results, label, color in [(results_racket, 'Racket', (0, 0, 255)), (results_shuttle, 'Shuttle', (255, 0, 0)), (results_net, 'Net', (0, 255, 0))]:
    bboxes = results.xyxy[0].cpu().numpy()  # Move to CPU and convert to numpy
    for bbox in bboxes:
        x1, y1, x2, y2, confidence, class_id = bbox[:6]
        # Draw bounding box on the original image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, f'{label} ({confidence:.2f})', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Crop and save the detected object
        cropped_object = image[int(y1):int(y2), int(x1):int(x2)]
        crop_path = f'C:\\Users\\rohit\\OneDrive\\Desktop\\net_kill_device\\badminton_net_fault_detection\\yolov5\\crops\\{label}_{count}.jpg'
        cv2.imwrite(crop_path, cropped_object)
        count += 1

# Save the image with all detections
output_image_path = r'C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\output_detections.jpg'
cv2.imwrite(output_image_path, image)

# Display the image with detections
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
