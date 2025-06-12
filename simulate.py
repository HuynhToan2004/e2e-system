import requests
import time
import random
import os

URL = "http://localhost:8070/predict"
IMG_DIR = "car_demo"

images = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for i in range(400):
    try:
        # Chọn ngẫu nhiên một ảnh
        img_path = random.choice(images)
        with open(img_path, "rb") as f:
            files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
            response = requests.post(URL, files=files)
            print(f"[{i+1}] Sent images → Status: {response.status_code}")
    except Exception as e:
        print(f"[{i+1}] Error sending image: {e}")
    
    time.sleep(3)
