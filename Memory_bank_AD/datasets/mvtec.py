import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

CLASS_NAMES = ["PZ1"]

class MVTecDataset(Dataset):
    def __init__(self, class_name, is_train=True, root="./data"):
        self.class_name = class_name
        self.is_train = is_train
        self.root = root
        self.img_paths = []
        self.labels = []
        self.transform = transforms.ToTensor()  # converte PIL in tensore

        base_dir = os.path.join(root, class_name)
        if is_train:
            img_dir = os.path.join(base_dir, "train")
            # immagini buone (label=0)
            for fname in os.listdir(img_dir):
                if fname.endswith(".png") or fname.endswith(".jpg"):
                    self.img_paths.append(os.path.join(img_dir, fname))
                    self.labels.append(0)
        else:
            test_dir = os.path.join(base_dir, "test")
            # cerca sottocartelle 'good' e 'defect'
            for subdir in os.listdir(test_dir):
                sub_path = os.path.join(test_dir, subdir)
                if os.path.isdir(sub_path):
                    label = 0 if subdir == 'good' else 1
                    for fname in os.listdir(sub_path):
                        if fname.endswith(".png") or fname.endswith(".jpg"):
                            self.img_paths.append(os.path.join(sub_path, fname))
                            self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)  # converte in tensore
        label = self.labels[idx]
        # maschera fittizia: tutto zero per buone, tutto uno per difettose
        if label == 0:
            mask = torch.zeros((224, 224), dtype=torch.uint8)
        else:
            mask = torch.ones((224, 224), dtype=torch.uint8)
        return img, label, mask