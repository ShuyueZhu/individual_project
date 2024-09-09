import numpy as np
import copy
import random

import torch.optim.lr_scheduler
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.utils.data import Subset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from dataset import *
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 3. 定义ResNet-34模型
def create_model():
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 修改最后一层，适应二分类

    return model.to(device)


# 4. 定义超参数空间
# param_grid = {
#     'learning_rate': [0.001, 0.005, 0.01],
#     'batch_size': [4, 8],
#     'optimizer': [optim.SGD, optim.Adam]
# }


# 5. 定义训练和验证函数
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0

        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        print(
            f'Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} '
            f'LR: {optimizer.param_groups[0]["lr"]: .8f}')

        # schedule.step()

        # 评估模型
        # model.eval()
        acc, f1, tpr = evaluate_model(model, test_loader)

        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Best model updated at epoch", epoch)

    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    # 提取混淆矩阵中的元素
    tn, fp, fn, tp = cm.ravel()

    # 计算 True Positive Rate (TPR) 即 Recall
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

    return accuracy, f1, tpr


# 6. 网格搜索
# best_acc = 0.0
# best_f1 = 0.0
# best_params = {}
#
# for lr in param_grid['learning_rate']:
#     for batch_size in param_grid['batch_size']:
#         for optim_func in param_grid['optimizer']:
#             print(f"Training with lr={lr}, batch_size={batch_size}, optimizer={optim_func.__name__}")
#
#             model = create_model()
#             criterion = nn.CrossEntropyLoss()
#             optimizer = optim_func(model.parameters(), lr=lr)
#
#             # 使用指定的batch_size创建train_loader和test_loader
#             train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#             test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#
#             # 训练模型
#             model = train_model(model, criterion, optimizer, train_loader, num_epochs=30, patience=5)
#
#             # torch.save(model.state_dict(), 'model_real.pth')
#             # 在测试集上评估模型
#             acc, f1 = evaluate_model(model, test_loader)
#
#             print(f"Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
#
#             if acc > best_acc:
#                 best_acc = acc
#                 best_f1 = f1
#                 best_params = {'learning_rate': lr, 'batch_size': batch_size, 'optimizer': optim_func}
#
# print(f"Best Parameters: {best_params}")
# print(f"Best Validation Accuracy: {best_acc:.4f}")
# print(f"Best Validation F1 Score: {best_f1:.4f}")

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_everything(316)

    root_dir = "D:\\ic\\binary"
    synthetic_root_dir = 'D:\\ic\\results'
    disease_label_dir = "disease"
    healthy_label_dir = "healthy"

    tfm = transforms.Compose([
        # normalize
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    disease_dataset = Mydata(root_dir, disease_label_dir, tfm)
    healthy_dataset = Mydata(root_dir, healthy_label_dir, tfm)
    synthetic_disease_dataset = Mydata(synthetic_root_dir, disease_label_dir, tfm)
    synthetic_healthy_dataset = Mydata(synthetic_root_dir, healthy_label_dir, tfm)

    # shuffle
    disease_dataset = Subset(disease_dataset, np.random.permutation(len(disease_dataset)))

    healthy_dataset = Subset(healthy_dataset, np.random.permutation(len(healthy_dataset)))

    # 提取 indices
    disease_indices = disease_dataset.indices
    healthy_indices = healthy_dataset.indices
    # 保存 indices 到 .txt 文件
    np.savetxt('disease_indices.txt', disease_indices, fmt='%d')
    np.savetxt('healthy_indices.txt', healthy_indices, fmt='%d')

    synthetic_disease_dataset = Subset(synthetic_disease_dataset, np.random.permutation(len(synthetic_disease_dataset)))
    synthetic_healthy_dataset = Subset(synthetic_healthy_dataset, np.random.permutation(len(synthetic_healthy_dataset)))

    # 计算前 70% 和 30% 的数据量
    # healthy50,disease174
    train_disease_size = int(0.6 * len(disease_dataset))
    train_healthy_size = int(0.6 * len(healthy_dataset))

    val_disease_size = int(0.2 * len(disease_dataset))
    val_healthy_size = int(0.2 * len(healthy_dataset))

    test_disease_size = len(disease_dataset) - train_disease_size - val_disease_size
    test_healthy_size = len(healthy_dataset) - train_healthy_size - val_healthy_size

    # 获取训练集，验证集和测试集的索引
    train_disease_indices = list(range(train_disease_size))
    train_healthy_indices = list(range(train_healthy_size))

    val_disease_indices = list(range(train_disease_size, train_disease_size + val_disease_size))
    val_healthy_indices = list(range(train_healthy_size, train_healthy_size + val_healthy_size))

    test_disease_indices = list(range(train_disease_size + val_disease_size, len(disease_dataset)))
    test_healthy_indices = list(range(train_healthy_size + val_healthy_size, len(healthy_dataset)))

    # 创建子集
    train_disease_dataset = Subset(disease_dataset, train_disease_indices)
    train_healthy_dataset = Subset(healthy_dataset, train_healthy_indices)
    print(len(train_disease_dataset))
    print(len(train_healthy_dataset))

    val_disease_dataset = Subset(disease_dataset, val_disease_indices)
    val_healthy_dataset = Subset(healthy_dataset, val_healthy_indices)

    test_disease_dataset = Subset(disease_dataset, test_disease_indices)
    test_healthy_dataset = Subset(healthy_dataset, test_healthy_indices)
    # print(type(real_disease_dataset))
    # 190train,34test   0.85
    train_dataset = train_disease_dataset + train_healthy_dataset
    val_dataset = val_disease_dataset + val_healthy_dataset
    test_dataset = test_disease_dataset + test_healthy_dataset

    # 添加特定比例的合成数据
    ratio_healthy = 1
    ratio_disease = 174 / 50 * ratio_healthy
    if ratio_disease or ratio_healthy:
        synthetic_train_disease_size = int(ratio_disease * train_disease_size) if int(
            ratio_disease * train_disease_size) < len(
            synthetic_disease_dataset) else len(synthetic_disease_dataset)
        synthetic_train_healthy_size = int(ratio_healthy * train_healthy_size) if int(
            ratio_healthy * train_healthy_size) < len(
            synthetic_healthy_dataset) else len(synthetic_healthy_dataset)

        synthetic_disease_indices = np.random.choice(len(synthetic_disease_dataset), synthetic_train_disease_size,
                                                     replace=False)
        synthetic_healthy_indices = np.random.choice(len(synthetic_healthy_dataset), synthetic_train_healthy_size,
                                                     replace=False)

        synthetic_disease_dataset = Subset(synthetic_disease_dataset, synthetic_disease_indices)
        synthetic_healthy_dataset = Subset(synthetic_healthy_dataset, synthetic_healthy_indices)

    if ratio_disease or ratio_healthy:
        train_dataset += synthetic_disease_dataset + synthetic_healthy_dataset
    # print(len(train_dataset))
    # print(len(val_dataset))
    # print(len(test_dataset))

    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    # schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # optimizer = optim_func(model.parameters(), lr=lr)

    # 使用指定的batch_size创建train_loader和test_loader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 训练模型
    model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)
    # torch.save(model.state_dict(), 'model_real.pth')
    torch.save(model.state_dict(), 'model_synthetic{}.pth'.format(ratio_healthy))
    # 在测试集上评估模型
    acc, f1, tpr = evaluate_model(model, test_loader)

    print(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    with open("add_synthetic.txt", "a", encoding="utf-8") as file:
        file.write(
            "Add {} times synthetic data\n Test Accuracy:{}\n F1 Score:{}\n TPR:{}\n".format(
                ratio_healthy, acc, f1, tpr))
