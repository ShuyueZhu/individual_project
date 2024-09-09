import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class Mydata(Dataset):

    def __init__(self, root_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        self.transform = transform
        # 创建从文件名到标签的映射
        if self.label_dir == "disease":
            self.labels = [0] * len(self.img_path)
        else:
            self.labels = [1] * len(self.img_path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        # 获取该图像的标签
        label = self.labels[idx]

        # 应用转换，将图像转换为 tensor
        if self.transform:
            img = self.transform(img)

        # 将标签转换为 tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img, label_tensor

    def __len__(self):
        return len(self.img_path)
