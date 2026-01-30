from roboflow import Roboflow
import os

API_KEY = "l1OJ14Ls16tkbxIdpMUY"

WORKSPACE = "pokmenon"
PROJECT = "pokemon_detection_scratch"
VERSION = 1

def download_dataset():
    rf = Roboflow(api_key=API_KEY)
    print("Loading Roboflow workspace...")
    
    project = rf.workspace(WORKSPACE).project(PROJECT)
    print("Loading Roboflow project...")

    dataset = project.version(VERSION).download(
        "yolov8",
        location="dataset"   # 👈 saves inside your project folder
    )

    print("✅ Dataset downloaded successfully!")
    print("📁 Saved at:", os.path.abspath("dataset"))

if __name__ == "__main__":
    download_dataset()
