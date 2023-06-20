import os
import random

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import DatasetFolder

# 设置随机种子，以便结果可复现
torch.manual_seed(42)

# 数据集路径
data_dir = 'data'

# 定义转换器
transform = transforms.Compose([
    transforms.Resize((200, 150)),  # 调整图像大小为200x150
    transforms.ToTensor()  # 将图像转换为张量
])

batch_size = 64


def image_list(imageRoot, txt='list.txt'):
    f = open(txt, 'wt')
    for (label, filename) in enumerate(sorted(os.listdir(imageRoot), reverse=False)):
        if os.path.isdir(os.path.join(imageRoot, filename)):
            for imagename in os.listdir(os.path.join(imageRoot, filename)):
                name, ext = os.path.splitext(imagename)
                ext = ext[1:]
                if ext == 'jpg' or ext == 'png' or ext == 'bmp':
                    f.write('%s %d\n' % (os.path.join(
                        imageRoot, filename, imagename), label))
    f.close()


def shuffle_split(listFile, trainFile, valFile):
    with open(listFile, 'r') as f:
        records = f.readlines()
    random.shuffle(records)
    num = len(records)
    trainNum = int(num * 0.8)
    with open(trainFile, 'w') as f:
        f.writelines(records[0:trainNum])
    with open(valFile, 'w') as f1:
        f1.writelines(records[trainNum:])


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


os.makedirs('./output', exist_ok=True)
if True:  # not os.path.exists('output/total.txt'):
    image_list(data_dir, 'output/total.txt')
    shuffle_split('output/total.txt', 'output/train.txt', 'output/val.txt')

train_data = MyDataset(txt='output/train.txt', transform=transforms.ToTensor())
val_data = MyDataset(txt='output/val.txt', transform=transforms.ToTensor())
train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size)

# 定义 ResNet 网络模型
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 输出10个类别（2个窗户开关状态 * 5个百叶窗开合大小）

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将模型移至 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 训练和验证循环
num_epochs = 100

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += torch.sum(predictions == labels.data)

    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = 100.0 * train_correct / len(train_loader.dataset)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_correct += torch.sum(predictions == labels.data)

    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = 100.0 * val_correct / len(val_loader.dataset)

    # 打印训练和验证结果
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}% - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%')
