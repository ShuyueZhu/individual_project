import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
from dataset import *

# 数据增强变换
transform_augmentation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_original = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 自定义数据集类



# 特征提取
# 定义函数以提取特征
def get_features(dataset, model, batch_size=4, device=device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []
    model = model.to(device)
    with torch.no_grad():
        for batch in loader:
            # 假设 batch 是一个元组，包含 (images, labels)
            if isinstance(batch, (tuple, list)):
                batch = batch[0]  # 只使用图像部分，不用标签
            batch = batch.to(device)
            feature = model(batch)
            features.append(feature.cpu().numpy())
    return np.concatenate(features, axis=0)


# 定义函数以计算 FID
def calculate_fid(real_features, generated_features):
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)

    return fid


# 加载 Inception v3 模型

model = models.inception_v3(pretrained=True, transform_input=False)
model = model.to(device)
model.eval()

# 加载数据集
root_dir = "D:\\ic\\binary"
synthetic_dir = "D:\\ic\\images"
disease_label_dir = "disease"
healthy_label_dir = "healthy"

original_disease_dataset = Mydata(root_dir, disease_label_dir, transform_original)
original_healthy_dataset = Mydata(root_dir, healthy_label_dir, transform_original)
# original_dataset = original_disease_dataset + original_healthy_dataset

synthetic_disease_dataset = Mydata(synthetic_dir, disease_label_dir, transform_original)
synthetic_healthy_dataset = Mydata(synthetic_dir, healthy_label_dir, transform_original)

# augmented_disease_dataset = Mydata(root_dir, disease_label_dir, transform_augmentation)
# augmented_healthy_dataset = Mydata(root_dir, healthy_label_dir, transform_augmentation)
# augmented_dataset = augmented_disease_dataset + augmented_healthy_dataset

# original_loader = DataLoader(original_dataset, batch_size=4, shuffle=False, num_workers=0)
# augmented_loader = DataLoader(augmented_dataset, batch_size=4, shuffle=False, num_workers=0)

# 提取特征
real_disease_features = get_features(original_disease_dataset, model)
# augmented_disease_features = get_features(augmented_disease_dataset, model)
synthetic_disease_features = get_features(synthetic_disease_dataset, model)
real_healthy_features = get_features(original_healthy_dataset, model)
# augmented_healthy_features = get_features(augmented_healthy_dataset, model)
synthetic_healthy_features = get_features(synthetic_healthy_dataset, model)

# 计算 FID
# fid_disease = calculate_fid(real_disease_features, augmented_disease_features)
# fid_healthy = calculate_fid(real_healthy_features, augmented_healthy_features)

fid_disease = calculate_fid(real_disease_features, synthetic_disease_features)
fid_healthy = calculate_fid(real_healthy_features, synthetic_healthy_features)

print(f"FID for Disease Leaves: {fid_disease}")
print(f"FID for Healthy Leaves: {fid_healthy}")

with open("FID.txt", "a", encoding="utf-8") as file:
    file.write(
        "Synthetic using CycleGAN\nFID for Disease Leaves:{}\nFID for Healthy Leaves:{}\n".format(
            fid_disease, fid_healthy))
