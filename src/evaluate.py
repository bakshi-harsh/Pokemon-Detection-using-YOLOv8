from ultralytics import YOLO

MODEL_PATH = "runs/pokemon_yolov8_best/weights/best.pt"

model = YOLO(MODEL_PATH)

metrics = model.val(data="dataset/data.yaml")

print("🎯 Evaluation Results")
print(f"mAP@50     : {metrics.box.map50}")
print(f"mAP@50-95  : {metrics.box.map}")
print(f"Precision : {metrics.box.mp}")
print(f"Recall    : {metrics.box.mr}")
