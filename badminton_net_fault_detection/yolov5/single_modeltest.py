from yolov5 import detect

detect.run(
    weights="C:/Users/rohit/OneDrive/Desktop/net_kill_device/badminton_net_fault_detection/yolov5/runs/train/prev/racket/weights/best.pt",
    source="C:/Users/rohit/OneDrive/Desktop/net_kill_device/badminton_net_fault_detection/yolov5/data/WhatsAppvideo.mp4",  # Your video path
    conf_thres=0.4,
    save_txt=True,
    save_conf=True,
    save=True  # ✅ This will save both images & videos
)
