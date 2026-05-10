import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, data_dir, is_train=True, transform=None):
        """
        AT&T Face Dataset 讀取器
        :param data_dir: 資料夾路徑，例如 '../dataset/original'
        :param is_train: True 表示訓練集 (前 8 張)，False 表示測試集 (後 2 張)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # AT&T dataset 有 person_1 到 person_40
        for person_name in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            # 從資料夾名稱 'person_X' 提取標籤 (0-39)
            label = int(person_name.split('_')[1]) - 1
            
            # 取得該人所有的圖片檔案並排序 (1.png ~ 10.png)
            images = [f for f in os.listdir(person_dir) if f.endswith('.png')]
            images.sort(key=lambda x: int(x.split('.')[0]))
            
            # 分割 Train (1-8) 和 Test (9-10)
            if is_train:
                selected_images = images[:8]
            else:
                selected_images = images[8:]
                
            for img_name in selected_images:
                self.image_paths.append(os.path.join(person_dir, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L') # 轉為灰階
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(base_dir="../dataset", batch_size=16):
    """
    取得所有需要的 DataLoader
    回傳:
      train_loader: 使用 Original 資料訓練
      test_loaders_dict: 包含各種測試資料的 DataLoader 字典
    """
    # 影像轉換 (轉成 Tensor 並標準化)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 1. 訓練用 DataLoader (只用 Original)
    train_dataset = FaceDataset(os.path.join(base_dir, 'original'), is_train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. 測試用 DataLoaders 字典
    test_loaders = {}
    
    # 需要測試的所有資料集名稱
    test_dirs = [
        'original', 
        'pixel_b4', 'pixel_b8', 'pixel_b16', 
        'blur_k15', 'blur_k45', 'blur_k99',
        'dp_pixel_b16_eps0.1', 'dp_pixel_b16_eps0.5', 'dp_pixel_b16_eps1.0',
        'dp_blur_k99_eps0.1', 'dp_blur_k99_eps0.5', 'dp_blur_k99_eps1.0'
    ]
    
    for d in test_dirs:
        dir_path = os.path.join(base_dir, d)
        if os.path.exists(dir_path):
            dataset = FaceDataset(dir_path, is_train=False, transform=transform)
            test_loaders[d] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
    return train_loader, test_loaders
