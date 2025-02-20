import os
import yaml

def count_images_from_yaml(yaml_path):
    # Load the YAML file
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    # Extract the train and val paths from the YAML data
    train_path = data['train']
    val_path = data['val']

    # Count images in the training directory
    train_images = [f for f in os.listdir(train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_train_images = len(train_images)

    # Count images in the validation directory
    val_images = [f for f in os.listdir(val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_val_images = len(val_images)

    # Print the counts
    print(f"Number of training images: {num_train_images}")
    print(f"Number of validation images: {num_val_images}")

# Specify the path to your YAML file
yaml_file_path = r'C:\Users\rohit\OneDrive\Desktop\net_kill_device\badminton_net_fault_detection\yolov5\data\dataset.yaml'

# Call the function
count_images_from_yaml(yaml_file_path)
