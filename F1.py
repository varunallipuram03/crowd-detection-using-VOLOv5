import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from ultralytics import YOLO

# ✅ Load model
model = YOLO("runs/detect/fast_train3/weights/best.pt")

# ✅ Predict on validation images (or test)
img_dir = "C:/crowd_dataset/valid/images"
label_dir = "C:/crowd_dataset/valid/labels"

results = model.predict(source=img_dir, conf=0.25, save=True)

# ✅ Get true and predicted labels (binary: 0 - no person, 1 - person present)
true_labels = []
pred_labels = []

img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")])

for i, img_file in enumerate(img_files):
    # Ground Truth
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(label_dir, label_file)

    has_person = 0
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                if line.strip().startswith("0") or line.strip().startswith("1"):
                    has_person = 1
                    break
    true_labels.append(has_person)

    # Prediction
    result = results[i]
    if result.boxes and len(result.boxes) > 0:
        pred_labels.append(1)
    else:
        pred_labels.append(0)

# ✅ Classification report
print("\n📊 Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=["No Person", "Person"]))

# ✅ Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Person", "Person"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
