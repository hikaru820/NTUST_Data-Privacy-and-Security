import os
import cv2
import numpy as np

def apply_pixelization(img, block_size):
    """
    將圖片進行馬賽克處理 (Pixelization)
    :param block_size: b 值 (例如 4, 8, 16)
    """
    h, w = img.shape[:2]
    # 縮小
    small = cv2.resize(img, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    # 放大回原尺寸
    pixelized = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelized

def apply_gaussian_blur(img, k):
    """
    將圖片進行高斯模糊 (Gaussian Blur)
    :param k: kernel size (例如 15, 45, 99)
    """
    return cv2.GaussianBlur(img, (k, k), 0)

def generate_deidentified_datasets(base_dir="../dataset"):
    original_dir = os.path.join(base_dir, "original")
    
    # 定義要做的去識別化參數
    methods = {
        'pixel_b4': lambda x: apply_pixelization(x, 4),
        'pixel_b8': lambda x: apply_pixelization(x, 8),
        'pixel_b16': lambda x: apply_pixelization(x, 16),
        'blur_k15': lambda x: apply_gaussian_blur(x, 15),
        'blur_k45': lambda x: apply_gaussian_blur(x, 45),
        'blur_k99': lambda x: apply_gaussian_blur(x, 99),
    }
    
    for method_name, func in methods.items():
        out_dir = os.path.join(base_dir, method_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Generating {method_name} ...")
        
        for person in os.listdir(original_dir):
            person_in_dir = os.path.join(original_dir, person)
            if not os.path.isdir(person_in_dir):
                continue
            
            person_out_dir = os.path.join(out_dir, person)
            os.makedirs(person_out_dir, exist_ok=True)
            
            for img_name in os.listdir(person_in_dir):
                if not img_name.endswith(('.png', '.jpg', '.pgm')):
                    continue
                    
                img_path = os.path.join(person_in_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # 套用去識別化
                    processed_img = func(img)
                    
                    # 儲存
                    out_path = os.path.join(person_out_dir, img_name)
                    cv2.imwrite(out_path, processed_img)

if __name__ == "__main__":
    generate_deidentified_datasets()
    print("去識別化資料集 (Pixelization, Gaussian Blur) 產生完成！")
