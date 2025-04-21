import os
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

#  Load your trained model
model = YOLO("runs/detect/fast_train3/weights/best.pt")

#  Path to images (we use validation folder here)
img_path = "C:/crowd_dataset/valid/images"
img_files = sorted(os.listdir(img_path))

#  Run prediction
results = model.predict(source=img_path, conf=0.25)

#  Count people per image
frame_ids = []
people_counts = []
density_labels = []

for i, result in enumerate(results):
    count = len(result.boxes)
    frame_ids.append(i + 1)
    people_counts.append(count)

    if count <= 10:
        density_labels.append("Low")
    elif count <= 25:
        density_labels.append("Medium")
    else:
        density_labels.append("High")

#  Create DataFrame
df = pd.DataFrame({
    "Frame": frame_ids,
    "People_Count": people_counts,
    "Density": density_labels
})

#  Save as CSV (optional for report)
df.to_csv("crowd_density_analysis.csv", index=False)

#  Plot 1: Time (frame) vs People Count
plt.figure(figsize=(10, 5))
plt.plot(df["Frame"], df["People_Count"], marker='o', linestyle='-', color='blue')
plt.title("People Count per Frame")
plt.xlabel("Frame")
plt.ylabel("Number of People")
plt.grid(True)
plt.savefig("people_count_plot.png")
plt.show()

#  Plot 2: Pie Chart of Density Classification
density_distribution = df["Density"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(density_distribution, labels=density_distribution.index, autopct="%1.1f%%", startangle=90, colors=["green", "orange", "red"])
plt.title("Crowd Density Distribution")
plt.savefig("density_pie_chart.png")
plt.show()

print(" Crowd density analysis complete. Graphs saved.")
