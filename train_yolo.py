from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # use 'yolov8s.pt' for better results if you want

# Train the model
model.train(
    data="crowd_dataset/data.yaml",  # 🟡 update with your actual path if it's different
    epochs=50,
    imgsz=640,
    batch=16,
    name="crowd_model"
)
