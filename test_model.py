import torch
import os

# Path to the model
model_path = 'models/6.pth'

# Verify file exists
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found in {os.getcwd()}!")
    exit()

try:
    # Load model, mapping to CPU
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")