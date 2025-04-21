import os

#  Step 0: Fix for OpenMP crash on CPU (must be set before importing torch or ultralytics)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#  Step 1: Clean label files to remove incorrect or unwanted labels
# Only keep labels that:
# - Have correct format (5 elements: class_id, x_center, y_center, width, height)
# - Belong to class 0 or class 1 (ignore other classes or segmentation data)

label_dirs = [
    "C:/crowd_dataset/train/labels",
    "C:/crowd_dataset/valid/labels"
]

for label_dir in label_dirs:
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            path = os.path.join(label_dir, filename)

            # Read the contents of the label file
            with open(path, "r") as file:
                lines = file.readlines()

            cleaned_lines = []
            for line in lines:
                parts = line.strip().split()

                if len(parts) != 5:
                    continue  # Skip if not a standard YOLO bbox format

                class_id = int(parts[0])
                if class_id > 1:
                    continue  # Keep only class 0 or 1

                cleaned_lines.append(" ".join(parts))

            # Overwrite the file with cleaned data
            with open(path, "w") as file:
                file.write("\n".join(cleaned_lines))

print(" Label files cleaned successfully.")

#  Step 2: Train YOLOv8n model with custom dataset
from ultralytics import YOLO

# Load pre-trained YOLOv8 nano model (small and fast)
model = YOLO("yolov8n.pt")

# Train the model on our dataset
model.train(
    data="C:/crowd_dataset/data.yaml",  # YAML file with dataset paths and class info
    epochs=10,             # Reduce for faster testing
    imgsz=640,             # Image size
    batch=8,               # Batch size for CPU
    device="cpu",          # CPU-only training
    name="fast_train"      # Custom name for this training run
)

print(" Training completed.")

#  Step 3: Predict on test images and save results
# After training, test the model on unseen images to see detections
results = model.predict(
    source="C:/crowd_dataset/test/images",  # Path to test images
    conf=0.25,      # Confidence threshold to filter weak detections
    save=True       # Save output images with bounding boxes
)

print(" Prediction completed and saved.")
