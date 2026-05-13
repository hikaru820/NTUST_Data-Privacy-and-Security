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

def get_dataloaders(base_dir="../dataset", batch_size=16, train_source='original'):
    """
    :param train_source: 訓練用 dataset 資料夾名稱
                         例如 'original'、'pixel_b16'、'blur_k99'
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 訓練 loader（用指定來源的 images 1–8）
    train_dir = os.path.join(base_dir, train_source)
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"找不到訓練資料夾: {train_dir}")
    train_dataset = FaceDataset(train_dir, is_train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 測試 loaders 維持自動掃描（images 9–10）
    test_loaders = {}
    for d in sorted(os.listdir(base_dir)):
        dir_path = os.path.join(base_dir, d)
        if not os.path.isdir(dir_path):
            continue
        dataset = FaceDataset(dir_path, is_train=False, transform=transform)
        if len(dataset) == 0:
            continue
        test_loaders[d] = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loaders