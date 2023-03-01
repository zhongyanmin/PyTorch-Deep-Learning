import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path


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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(   # standardized processing
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

train_path = './kaggle/inputs/train'
test_path = './kaggle/inputs/test'
# train_dataset = datasets.ImageFolder('./kaggle/inputs/train', transform=transform)
# test_dataset = datasets.ImageFolder('./kaggle/inputs/test', transform=transform)
train_dataset = get_dataset(train_path, transform=transform)
test_dataset = get_dataset(test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(-1, 64 * 28 * 28)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.softmax(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

net.to(device)
epochs = 25
train_loss_list = []
train_acc_list = []

for epoch in range(epochs):
    print("Epoch {} / {}".format(epoch + 1, epochs))
            
    t_loss, t_corr = 0.0, 0.0
    
    net.train() 
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        preds = net(inputs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * inputs.size(0)
        t_corr += torch.sum(preds.argmax(1) == labels) 
        
    train_loss = t_loss / len(train_loader.dataset)
    train_acc = t_corr.cpu().numpy() / len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)  
    print('Train Loss: {:.4f} Accuracy: {:.4f}%'.format(train_loss, train_acc * 100))
    
plt.figure()
plt.title('Train Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('')
plt.plot(range(1, epochs+1), np.array(train_loss_list), color='blue',
         linestyle='-', label='Train_Loss')
plt.plot(range(1, epochs+1), np.array(train_acc_list), color='red',
         linestyle='-', label='Train_Accuracy')
plt.legend()  # 凡例
plt.show()  # 表示


v_loss,  v_corr = 0.0, 0.0      
net.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        preds = net(inputs)
        v_loss += loss.item() * inputs.size(0)
        v_corr += torch.sum(preds.argmax(1) == labels)
        
    print('Test Loss: {:.4f} Accuracy: {:.4f}%'.format(v_loss / len(test_loader.dataset),
                                                       (v_corr / len(test_loader.dataset)) * 100))