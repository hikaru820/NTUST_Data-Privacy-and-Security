import os
import cv2
import numpy as np

def add_laplace_noise(img, epsilon, b=None, k=None, m=16):
    """
    對已 pixelized 或 blurred 過的圖加 Laplace noise (DP-Pix / DP-Blur)
    :param b: 若來源是 pixelization，傳入 block size
    :param k: 若來源是 Gaussian blur，傳入 kernel size
    :param m: neighborhood 參數 (預設 16，跟隨 Fan TPDP2019)
    """
    if b is not None:
        sensitivity = 255.0 * m / (b * b)        # DP-Pix
    elif k is not None:
        sensitivity = 255.0 * m / (k * k)        # DP-Blur (近似)
    else:
        sensitivity = 255.0
    
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=img.shape)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img

def generate_dp_datasets(
    base_dir="../dataset",
    sources=None,
    epsilons=None,
    seed=42,
    overwrite=False,
    test_only=False,        # 如果為 True，則只對編號 9-10 的圖片加噪（FaceDataset 實際使用的測試集）
):
    # 預設值：定義要處理的原始來源與 ε（Epsilon）的網格組合
    if sources is None:
        sources = ['original', 'pixel_b4', 'blur_k15', 'pixel_b16', 'blur_k99']
    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]

    # 確保可重複性 — 讓每次執行產生的雜訊模式相同
    rng = np.random.default_rng(seed)

    # 根據「此腳本檔案的位置」來解析 base_dir，而非執行時的當前目錄 (cwd)
    # 這能避免從不同資料夾執行程式時噴出 FileNotFoundError
    if not os.path.isabs(base_dir):
        here = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.normpath(os.path.join(here, base_dir))

    total = len(sources) * len(epsilons)
    done = 0

    for source in sources:
        source_dir = os.path.join(base_dir, source)
        if not os.path.isdir(source_dir):
            print(f"[跳過] 找不到來源路徑: {source_dir} (是否尚未執行 deidentify.py?)")
            done += len(epsilons)
            continue

        for eps in epsilons:
            done += 1
            # 強制轉換為浮點數格式，例如將 100 轉為 'eps100.0'，以符合現有的資料夾命名規則
            eps_str = f"{float(eps)}"
            dp_method_name = f"dp_{source}_eps{eps_str}"
            out_dir = os.path.join(base_dir, dp_method_name)

            # 除非明確要求覆蓋 (overwrite)，否則跳過已存在的資料夾
            if (not overwrite) and os.path.isdir(out_dir) \
               and any(os.scandir(out_dir)):
                print(f"[{done}/{total}] {dp_method_name} (資料夾已存在，跳過)")
                continue

            os.makedirs(out_dir, exist_ok=True)
            print(f"[{done}/{total}] 正在產生 {dp_method_name} ...")

            for person in os.listdir(source_dir):
                person_in_dir = os.path.join(source_dir, person)
                if not os.path.isdir(person_in_dir):
                    continue
                person_out_dir = os.path.join(out_dir, person)
                os.makedirs(person_out_dir, exist_ok=True)

                for img_name in os.listdir(person_in_dir):
                    if not img_name.lower().endswith(('.png', '.jpg', '.pgm')):
                        continue

                    # 選擇性加速：只對測試中實際會用到的圖片加雜訊
                    if test_only:
                        try:
                            idx = int(os.path.splitext(img_name)[0])
                            if idx not in (9, 10):
                                continue
                        except ValueError:
                            pass

                    img_path = os.path.join(person_in_dir, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    # 使用已設定種子 (seed) 的 RNG 取代 np.random.laplace，確保結果可復現
                    # 根據差分隱私公式，scale 設為 255.0 / eps [cite: 79, 107]
                    scale = 255.0 / eps
                    noise = rng.laplace(loc=0.0, scale=scale, size=img.shape)
                    
                    # 將雜訊加入影像，並將數值限制在 0-255 之間，最後轉回 uint8 格式
                    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

                    cv2.imwrite(os.path.join(person_out_dir, img_name), noisy)

if __name__ == "__main__":
    generate_dp_datasets()
    print("Differential Privacy 資料集產生完成！")
