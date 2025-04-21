# 📊 Crowd Density Estimation using Video Analytics for Smart City Monitoring

This project aims to develop a smart city application that uses video analytics and deep learning to detect and estimate crowd density in public areas using real-world footage. It utilizes the **YOLOv8 object detection model** to detect people in images and evaluate crowd levels.

---

## 🧠 Objective

To design a computer vision system that:
- Detects people in crowd scenes
- Estimates the density of the crowd
- Visualizes predictions
- Evaluates performance using standard classification metrics (F1-score, precision, recall, etc.)

---

## 📁 Folder Structure

```
crowd-density-estimation/
│
├── images/                   # Input images extracted from real-world videos
├── labels/                   # YOLO format annotations for training
├── runs/detect/train/        # YOLOv8 training logs and weights
├── results/                  # Predicted images and person counts
├── dataset.yaml              # Dataset config for YOLOv8
├── training_script.py        # Training code
├── inference_script.py       # Inference code
├── evaluation_metrics.ipynb  # Notebook to compute F1, precision, recall
└── README.md                 # This file
```

---

## 🔧 Requirements

Install dependencies using pip:

```bash
pip install ultralytics matplotlib opencv-python numpy pandas
```

Also, ensure:
- Python ≥ 3.8
- CUDA enabled GPU (optional but faster training)
- OpenMP conflict fixed (set `KMP_DUPLICATE_LIB_OK=TRUE` if needed on CPU)

---

## 📷 Step 1: Data Collection

- A **real-world video** (e.g. from WhatsApp or mobile camera) was used to capture crowd scenes.
- Frames were extracted and saved as individual `.jpg` images in the `images/` folder.

---

## 🏷️ Step 2: Image Annotation

- Used tools like **Roboflow** or **LabelImg** to annotate each image manually.
- Every person in the image is annotated using bounding boxes.
- Labels are saved in **YOLO format** inside the `labels/` folder.
  
Each `.txt` file looks like:
```
0 0.52 0.66 0.23 0.36
0 0.34 0.55 0.18 0.24
...
```

---

## 🧠 Step 3: Model Training with YOLOv8

Trained YOLOv8 on the custom crowd dataset using:

```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=30 imgsz=640 batch=4
```

The `dataset.yaml` config:

```yaml
train: images/train
val: images/val
nc: 1
names: ['person']
```

Model outputs are saved in `runs/detect/train/`.

---

## 🔍 Step 4: Inference (Testing)

Run inference using the trained model:

```bash
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=images/test save=True
```

This generates:
- Predicted images with bounding boxes
- Person counts per image
- Output stored in `runs/detect/predict/`

---

## 📈 Step 5: Evaluation Metrics

Compared prediction results with ground truth to evaluate model:

- Precision
- Recall
- F1 Score
- Confusion Matrix

Evaluation was done using a custom Jupyter Notebook: `evaluation_metrics.ipynb`

Example outputs:
```
Precision: 0.88
Recall: 0.83
F1 Score: 0.85
```

Confusion Matrix:

|          | Predicted Positive | Predicted Negative |
|----------|--------------------|--------------------|
| Actual Positive | TP = 50         | FN = 10         |
| Actual Negative | FP = 7          | TN = 33         |

---

## 👁️ Step 6: Visualization

- Annotated predictions with bounding boxes
- Showed total person count per image
- Optionally calculated **density = count / area**

Used matplotlib and OpenCV to visualize and save results in `results/` folder.

---

## 📌 Applications

- Real-time crowd monitoring in public areas
- Safety alerts for overcrowding
- Smart city planning and resource allocation
- Event management systems

---

## 🧾 Conclusion

This project successfully demonstrates how deep learning models like YOLOv8 can be applied to detect and estimate crowd densities from real-world video data. With further tuning, it can be deployed in smart city monitoring systems for crowd control and public safety.

---

## 🙌 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Annotation Tool](https://roboflow.com/)
- [OpenCV](https://opencv.org/)

---

## 📬 Contact

**Author:** [VARUN VENKATA ALLIPURAM]  
**Email:** varunallipuram@gmail.com  


