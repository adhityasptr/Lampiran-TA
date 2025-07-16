from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Get detailed metrics
metrics = model.val()

# Print metrics
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")  
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")

# Get model info
import os
model_size = os.path.getsize('runs/detect/train/weights/best.pt') / (1024*1024)  # MB
print(f"Model size: {model_size:.1f} MB")