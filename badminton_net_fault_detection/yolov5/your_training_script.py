import torch
import os
from pathlib import Path
from yolov5.train import run   # Import the train function

def main():
    # Ensure only the RTX 3050 is visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Check if CUDA is available and get the current device
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Please check your GPU.")

    # Use the first available GPU
    device = 'cuda:0'  # This should point to your RTX 3050

    # Print available devices
    print(f"CUDA is available. Using device: {device}")
    print("Available CUDA devices:")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    # Define the path to your dataset YAML file
    dataset_yaml_path = Path('C:\\Users\\rohit\\OneDrive\\Desktop\\net_kill_device\\badminton_net_fault_detection\\yolov5\\data\\dataset.yaml')

    # Set up training parameters
    params = {
        'data': str(dataset_yaml_path),  # Use the correct path as a string
        'weights': 'yolov5s.pt',  # Path to your model weights
        'img': 640,
        'batch': 4,  # Lower batch size
        'epochs': 80,  # Fewer epochs for testing
        'cache': True,  # Enable caching
        'workers': 0,  # Reduce number of workers if facing memory issues
        'device': device,  # Ensure the device is set to your GPU
    }

    try:
        print("Starting training...")
        run(**params)  
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    main()
