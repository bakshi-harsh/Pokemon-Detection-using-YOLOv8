# from ultralytics import YOLO
# import torch

# def train_model():
#     # Check if GPU is available
#     device = 0 if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # Start training from pretrained weights
#     model = YOLO("yolov8s.pt")   # best balance of accuracy for 4GB GPU

#     model.train(
#         data="dataset/data.yaml",
#         epochs=120,              # high accuracy
#         imgsz=640,
#         batch=8,                 # safe for 4GB VRAM
#         lr0=0.001,
#         optimizer="AdamW",
#         patience=25,
#         augment=True,
#         device=device,
#         project="runs",
#         name="pokemon_yolov8_best",
#         workers=2,
#         save=True,
#         save_period=5            # saves checkpoint every 5 epochs
#     )

# if __name__ == "__main__":
#     train_model()
from ultralytics import YOLO
import torch

def train_model():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load last checkpoint
    model = YOLO("runs/pokemon_yolov8_best/weights/last.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=120,          # total epochs target
        imgsz=640,
        batch=8,
        lr0=0.001,
        optimizer="AdamW",
        patience=25,
        augment=True,
        device=device,
        project="runs",
        name="pokemon_yolov8_best",
        workers=2,
        resume=True          # THIS is what resumes training
    )

if __name__ == "__main__":
    train_model()
