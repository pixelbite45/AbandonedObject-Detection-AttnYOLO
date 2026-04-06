from ultralytics import YOLO
import torch

model = YOLO("yolo11n.pt")
print(f"Using device: {model.device}") # This should say 'cpu'