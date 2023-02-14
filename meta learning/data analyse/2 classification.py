import glob
import os

import cv2 as cv
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class NLDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        img_pths = glob.glob(img_dir + os.sep + '*.jpg')
        assert img_pths, 'no jpg file in ' + img_dir
        self.img_pths = img_pths

    def __len__(self):
        return len(self.img_pths)

    def __getitem__(self, idx):
        img_pth = self.img_pths[idx]
        img_name = img_pth.split(os.sep)[-1]

        label = 1 if 'N' in img_name.split('.')[0] else 0
        label = torch.tensor(label)
        image = cv.imread(img_pth)
        image = cv.resize(image, (360, 360), cv.INTER_LINEAR)  # resize
        image = image / 255.0  # 归一化
        image = torch.from_numpy(image)  # 转为Tensor
        image = image.permute(2, 0, 1).to(torch.float32)
        return image, label


from torch import nn
import torch.nn.functional as F


class NLNet(nn.Module):
    def __init__(self):
        super(NLNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bat1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=3, stride=1)
        self.bat2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=2)
        self.bat3 = nn.BatchNorm2d(384)
        self.fc1 = nn.Linear(384 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.bat1(self.conv1(x))
        x = self.pool(F.relu(self.bat1(x)))
        x = self.bat2(self.conv2(x))
        x = self.pool(F.relu(self.bat2(x)))
        x = self.bat3(self.conv3(x))
        x = self.pool(F.relu(self.bat3(x)))
        x = x.view(-1, 384 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


import time


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    t0 = time.time()
    for batch, (X, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch + 1) % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] ({time.time() - t0:.3f}s)")
            t0 = time.time()


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # 使用gpu加速
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 1. 使用DataLoader和自定义的Dataset加载数据
trainset = NLDataset('./data final2/train')
testset = NLDataset('./data final2/test')
train_loader = DataLoader(trainset, batch_size=16, num_workers=0, shuffle=True)
test_loader = DataLoader(testset, batch_size=16, num_workers=0, shuffle=True)

# 2. 创建模型，确定损失函数和优化函数
model = NLNet()
loss_fn = nn.CrossEntropyLoss()
# GPU 加速
if torch.cuda.is_available():
    model = model.cuda()
    loss_fn = loss_fn.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. 训练模型
epochs = 20
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)
print("Done!")
model = NLNet()
torch.save(model.state_dict, './pretrained model.pth')