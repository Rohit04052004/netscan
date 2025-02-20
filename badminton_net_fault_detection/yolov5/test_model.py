import subprocess
import torch
from pathlib import Path

# Define paths
venv_python_path = Path(r"C:\Users\rohit\OneDrive\Desktop\net_kill_device\.venv\Scripts\python.exe")
weights_path = r"C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\runs\train\racket\weights\best.pt"
image_path = r"C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\data\IMG_3304.jpg"

# Check if CUDA is available, else default to CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Command to run detect.py
command = [
    str(venv_python_path), 'detect.py',
    '--weights', weights_path,
    '--source', image_path,
    '--img', '640',
    '--conf-thres', '0.4',
    '--iou-thres', '0.5',
    '--device', 'cpu',  # Change to 'cpu'
    '--project', 'runs/detect',
    '--name', 'exp_test'
]

# Run the command
subprocess.run(command, check=True)
