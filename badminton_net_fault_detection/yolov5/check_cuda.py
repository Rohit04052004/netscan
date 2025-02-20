import torch

# Check if CUDA is available and get GPU details
if torch.cuda.is_available():
    print("CUDA is available. Using GPU:")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")
torch.cuda.empty_cache()
