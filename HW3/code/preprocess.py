import os
import cv2
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

def download_and_save_dataset(base_dir="../dataset/original"):
    """
    下載 AT&T Face Dataset (Olivetti Faces) 並儲存為圖片檔案。
    資料集共 40 人，每人 10 張，解析度為 64x64 (灰階)。
    將 8 張做為 train，2 張做為 test。
    """
    print("Downloading AT&T Face Dataset...")
    faces = fetch_olivetti_faces()
    images = faces.images  # (400, 64, 64)
    targets = faces.target  # (400,)
    
    os.makedirs(base_dir, exist_ok=True)
    
    for i in range(len(images)):
        person_id = targets[i] + 1  # 1 到 40
        img_id = (i % 10) + 1       # 1 到 10
        
        person_dir = os.path.join(base_dir, f"person_{person_id}")
        os.makedirs(person_dir, exist_ok=True)
        
        # sklearn 下載的影像數值為 [0, 1]，轉回 [0, 255]
        img_array = (images[i] * 255).astype(np.uint8)
        
        img_path = os.path.join(person_dir, f"{img_id}.png")
        cv2.imwrite(img_path, img_array)
        
    print(f"Dataset successfully saved to {base_dir}")
    print("總共 40 位受試者，每位 10 張圖片。")

if __name__ == "__main__":
    download_and_save_dataset()
