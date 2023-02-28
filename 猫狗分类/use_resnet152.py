# coding: utf-8
import time
from PIL import Image
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader,  random_split
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet152_Weights
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

get_label = lambda x: x.name.split('.')[0]


class get_dataset(Dataset):
    def __init__(self, root, transform=None):
        self.images = list(Path(root).glob('*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = get_label(img)
        label = 1 if label == 'dog' else 0
        if self.transform:
            img = self.transform(Image.open(img))
        return img, torch.tensor(label, dtype=torch.int64)


transforms = transforms.Compose([
    transforms.Resize([224, 224]),  # 将输入图片resize成统一尺寸
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
    transforms.Normalize(   # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])  # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计
])

# train_path = r'C:\Users\zhongyanmin\Downloads\train'
train_path = './kaggle/inputs/train'
# get dataset
dataset = get_dataset(train_path, transform=transforms)
print(len(dataset))
# splitting into train and validation
train_data, valid_data = random_split(
    dataset,
    lengths=[int(len(dataset) * 0.5),
             int(len(dataset) * 0.5)],
    generator=torch.Generator().manual_seed(7))
print("train: ", len(train_data))
print("valid: ", len(valid_data))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)

# set config
feature_extract = True
num_epochs = 20


class ResNet(nn.Module):
    def __init__(self, num_classes, feature_extract):
        super(ResNet, self).__init__()
        self.resnet = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        if feature_extract:
            for param in self.resnet.parameters():
                param.requires_grad = False
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(num_features, num_classes),
                                       nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.resnet(x)
        return x


model = ResNet(2, feature_extract)
# 是否训练所有层
params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# GPU计算
model = model.to(device)
# 优化器设置
optimizer = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7,
                                      gamma=0.1)  #学习率每7个epoch衰减成原来的1/10
#最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    since = time.time()
    t_loss, t_acc = 0.0, 0.0
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = torch.max(outputs, 1)[1]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t_loss += loss.item() * inputs.size(0)
        t_acc += torch.sum(preds == labels)

    time_elapsed = time.time() - since
    print('Time elapsed {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(t_loss / len(train_loader.dataset),
                                                  t_acc / len(train_loader.dataset)))

print("*" * 25, "start valid", "*" * 25)
v_loss, v_acc = 0.0, 0.0
since = time.time()
model.eval()
with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = torch.max(outputs, 1)[1]

        v_loss += loss.item() * inputs.size(0)
        v_acc += torch.sum(preds == labels)

    time_elapsed = time.time() - since
    print('Time elapsed {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
    print('Valid Loss: {:.4f} Acc: {:.4f}'.format(v_loss / len(valid_loader.dataset),
                                                    v_acc / len(valid_loader.dataset)))


print("*" * 25, "start test", "*" * 25)
# test_path = r'C:\Users\zhongyanmin\Downloads\test'
test_path = './kaggle/inputs/test'
# get dataset
test_dataset = get_dataset(test_path, transform=transforms)
test_loader = DataLoader(test_dataset, batch_size=32)
_loss, _acc = 0.0, 0.0
since = time.time()
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = torch.max(outputs, 1)[1]

        _loss += loss.item() * inputs.size(0)
        _acc += torch.sum(preds == labels)

    time_elapsed = time.time() - since
    print('Time elapsed {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
    print('Test Loss: {:.4f} Acc: {:.4f}'.format(_loss / len(test_loader.dataset),
                                                    _acc / len(test_loader.dataset)))


def im_convert(image):
    # 颜色通道还原
    image = np.array(image).transpose(1, 2, 0)
    # 预处理还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)

    return image


dataiter = iter((test_loader))
fig = plt.figure(figsize=(10, 8))
images, labels = next(dataiter)
model.eval()
output = model(images.cuda())
pred = torch.max(output, 1)[-1]

for idx in range(4 * 4):
    ax = plt.subplot(4, 4, idx + 1, xticks=[], yticks=[])
    ax.set_title('dog' if pred[idx] == 1 else 'cat', color=('blue' if pred[idx] == labels[idx] else 'red'))
    plt.imshow(im_convert(images[idx]))

plt.show()