from ultralytics import YOLO
import cv2
import os

# Path to your trained model
MODEL_PATH = "runs/pokemon_yolov8_best/weights/best.pt"

model = YOLO(MODEL_PATH)


def detect_image(image_path):
    if not os.path.exists(image_path):
        print("❌ Image file not found!")
        return

    results = model(image_path)
    annotated = results[0].plot()

    cv2.imshow("Pokemon Detection - Image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_video(video_path):
    if not os.path.exists(video_path):
        print("❌ Video file not found!")
        return

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        cv2.imshow("Pokemon Detection - Video", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        cv2.imshow("Pokemon Detection - Webcam", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\nChoose Detection Mode:")
    print("1 - Detect from Image")
    print("2 - Detect from Video")
    print("3 - Detect from Webcam")

    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        img_path = input("Enter image path: ")
        detect_image(img_path)

    elif choice == "2":
        video_path = input("Enter video path: ")
        detect_video(video_path)

    elif choice == "3":
        detect_webcam()

    else:
        print("❌ Invalid choice! Please select 1, 2, or 3.")
