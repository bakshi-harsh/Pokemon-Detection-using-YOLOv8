🧠 Pokémon Object Detection using YOLOv8

An end-to-end computer vision project that detects Pokémon in images, videos, and live webcam streams using a high-accuracy YOLOv8 model. The project follows a clean, production-ready ML pipeline including dataset management, model training, evaluation, and real-time inference.

🚀 Project Overview

This project aims to build a robust Pokémon object detection system using deep learning. A custom annotated dataset was sourced from Roboflow, trained using YOLOv8, and optimized for high accuracy on limited GPU resources. The final system supports inference on static images, video files, and live camera feeds.

✨ Key Features

📦 Custom annotated Pokémon dataset (Roboflow)
🔍 High-accuracy object detection using YOLOv8
🎥 Supports image, video, and webcam inference
⚡ Optimized training on 4GB GPU (RTX 2050)
📊 Model evaluation using mAP, precision, and recall
🧩 Modular and clean project architecture
🔁 Resume-safe training with checkpoint support


🗂️ Project Structure

Pokemon-Detection-using-YOLOv8/
│
├── src/
│   ├── download_dataset.py   # Roboflow API dataset download
│   ├── train.py              # Model training
│   ├── evaluate.py           # Model evaluation
│   └── test.py               # Image / Video / Webcam inference
│
├── dataset/                  # Downloaded dataset (YOLO format)
│   └── data.yaml
│
├── runs/
│   └── pokemon_yolov8_best/
│       └── weights/
│           └── best.pt       # Trained YOLOv8 model
│
├── screenshots/              # Training & detection outputs
├── requirements.txt
└── README.md

📊 Dataset Information

Source: Roboflow
Type: Object Detection
Format: YOLOv8
Annotations: Bounding boxes for multiple Pokémon classes
Split: Train / Validation / Test

🔗 Dataset Access

The dataset was sourced and managed using Roboflow.
You can access or recreate the dataset from Roboflow by searching for a Pokémon object detection dataset or by using your own Roboflow workspace and annotations.
To download via API, use:
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("pokmenon").project("pokemon_detection_scratch")
dataset = project.version(1).download("yolov8", location="dataset")

🛠️ Tech Stack

1. Language: Python
2. Model: YOLOv8 (Ultralytics)
3. Libraries: PyTorch, OpenCV, NumPy
4. Dataset Tool: Roboflow
5. Hardware: NVIDIA RTX 2050 (4GB VRAM)

⚙️ Installation

1. Clone the repository:
git clone https://github.com/bakshi-harsh/Pokemon-Detection-using-YOLOv8.git
cd Pokemon-Detection-using-YOLOv8

2. Install dependencies:
pip install -r requirements.txt

🏆 Results

1. Achieved strong detection performance on multiple Pokémon classes
2. Stable training with improved generalization
3. Real-time inference on webcam with smooth FPS
4. Screenshots and sample outputs are available in the screenshots/ folder.

📌 Use Cases

1. Computer Vision learning projects
2. Deep Learning portfolios
3. Object detection research
4. Internship / placement projects
5. Real-time vision applications

📄 License

This project is for educational and research purposes only.
Pokémon images and names belong to their respective owners.

👨‍💻 Author

Harsh Kumar
MCA (Hons. AI & ML)
GitHub: https://github.com/bakshi-harsh

