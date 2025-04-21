from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Or use 'yolov8s.pt' for a stronger model

# Train the model using your data.yaml file
model.train(
    data="path/to/data.yaml",  # 👈 Change this to your actual data.yaml path
    epochs=50,
    imgsz=640,
    batch=16,
    name="crowd_model"
)
