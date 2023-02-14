
import glob
import os

import cv2 as cv

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


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
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear
        nn.LogSoftmax(dim=1)

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


# Load the pre-trained model
model = NLNet()
model.load_state_dict(torch.load('./pretrained model.pth'))
model.eval()
num_classes = 2

# Freeze all layers


# Replace the last fully-connected layer
n_inputs = model.fc.in_features
model.fc4 = nn.Sequential(
    nn.Linear(n_inputs, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 2),
    nn.LogSoftmax(dim=1)
)

# Define loss function and optimizer
criterion= torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Load the data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.ImageFolder(root='data/training', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

# Plotting tools
loss_history = []

# Train the model
for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Record the loss
        running_loss += loss.item()
        loss_history.append(loss.item())

        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Plot the loss history
plt.plot(loss_history)
plt.xlabel('Batch Index')
plt.ylabel('Loss')
plt.show()
