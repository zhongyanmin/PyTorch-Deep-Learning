# coding: UTF-8
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import warnings
warnings.filterwarnings('ignore')


train_path = './data/train'
test_path = './data/test'

train_transforms = transforms.Compose([
    transforms.Resize([224, 224]),  # 将输入图片resize成统一尺寸
    # transforms.RandomRotation(degrees=(-10, 10)),  # 随机旋转，-10到10度之间随机选
    # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
    # transforms.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
    # transforms.RandomPerspective(distortion_scale=0.6, p=1.0), # 随机视角
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # 随机选择的高斯模糊模糊图像
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
    transforms.Normalize(   # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])  # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计
])

test_transforms = transforms.Compose([
    transforms.Resize([224, 224]),  # 将输入图片resize成统一尺寸
    transforms.ToTensor(),          # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
    transforms.Normalize(           # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])  # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计算得到的。
])

train_data = datasets.ImageFolder(train_path, transform=train_transforms)
test_data = datasets.ImageFolder(test_path, transform=test_transforms)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4)

for x, y in test_loader:
    print("Shape of x [N, C, H, W]: ", x.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# 设置为gpu训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 2)
        self.conv2 = nn.Conv2d(64, 128, 5, 2)
        self.fc1 = nn.Linear(128*13*13, 128)
        self.fc2 = nn.Linear(128, 2)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # x size: [3, 224, 224]
        x = F.relu(self.conv1(x))
        # x size: [64, 110, 110]
        x = self.pool(x)
        # x size: [64, 55, 55]
        x = F.relu(self.conv2(x))
        # x size: [128, 26, 26]
        x = self.pool(x)
        # x size: [128, 13, 13]
        x = x.view(-1, 128*13*13)
        # x size: [1, 128*13*13]
        x = F.relu(self.fc1(x))
        # x size: [1, 128]
        x = F.relu(self.fc2(x))
        # x size: [1, 2]
        x = F.log_softmax(x, dim=1)
        return x
    

model = CNN()
model.to(device)
# 损失函数
loss_function = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        preds = model(x)
        # 计算预测误差
        loss = loss_function(preds, y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"Train Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            test_loss += loss_fn(preds, y).item()
            correct += (preds.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
epochs = 20
for epoch in range(epochs):
    print("Epoch {} / {}".format(epoch + 1, epochs))
    train(train_loader, model, loss_function, optimizer)
    test(test_loader, model, loss_function)