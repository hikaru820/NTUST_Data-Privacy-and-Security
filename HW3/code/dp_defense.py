import os
import cv2
import numpy as np

def add_laplace_noise(img, epsilon, sensitivity=255.0):
    """
    對圖片加入 Laplace Noise 以達成 Differential Privacy
    :param img: 原始圖片 (0-255)
    :param epsilon: 隱私預算 (越小保護越強，雜訊越大)
    :param sensitivity: 影像像素最大可能變動值，預設為 255
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=img.shape)
    
    noisy_img = img.astype(np.float32) + noise
    
    # 限制數值範圍並轉換回 uint8
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def generate_dp_datasets(base_dir="../dataset"):
    # 根據 PDF，我們對 pixel_b16 和 blur_k99 加上 DP noise
    sources = ['pixel_b16', 'blur_k99']
    epsilons = [0.1, 0.5, 1.0]
    
    for source in sources:
        source_dir = os.path.join(base_dir, source)
        if not os.path.exists(source_dir):
            print(f"找不到來源資料夾: {source_dir}，請先執行 deidentify.py！")
            continue
            
        for eps in epsilons:
            dp_method_name = f"dp_{source}_eps{eps}"
            out_dir = os.path.join(base_dir, dp_method_name)
            os.makedirs(out_dir, exist_ok=True)
            print(f"Generating {dp_method_name} ...")
            
            for person in os.listdir(source_dir):
                person_in_dir = os.path.join(source_dir, person)
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
                        # 加入 DP Noise
                        noisy_img = add_laplace_noise(img, eps)
                        
                        # 儲存
                        out_path = os.path.join(person_out_dir, img_name)
                        cv2.imwrite(out_path, noisy_img)

if __name__ == "__main__":
    generate_dp_datasets()
    print("Differential Privacy 資料集產生完成！")
