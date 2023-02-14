import copy
import os
from io import BytesIO

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm

TRAIN_PATH = "./data/train/"
VAL_PATH = "./data/test/"
NUM_BATCH = 8
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
    ToTensor(),
    Resize((224, 224))
])


class CatDogDataset(Dataset):

    def __init__(self, train_dir, transform=None):
        self.train_dir = train_dir
        self.transform = transform
        self.images = os.listdir(train_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.train_dir, self.images[index])
        label = self.images[index].split(".")[0]

        label = 0 if 'N' in label else 1

        image = np.array(Image.open(image_path))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


train_data = CatDogDataset(TRAIN_PATH, transform)
val_data = CatDogDataset(VAL_PATH, transform)

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
model = NLNet()
model.load_state_dict(torch.load('./pretrained model.pth'))
for param in model.parameters():
    param.requires_grad = False

model.fc4 = nn.Sequential(*[
    nn.Linear(in_features=512, out_features=2),
    nn.Softmax(dim=1)
])
train_dl = DataLoader(train_data, batch_size=NUM_BATCH)
val_dl = DataLoader(val_data, batch_size=NUM_BATCH)




def validate(model, data):
    total = 0
    correct = 0

    for (images, labels) in data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        x = model(images)
        _, pred = torch.max(x, 1)
        total += x.size(0)
        correct += torch.sum(pred == labels)

    return correct * 100 / total


def train(num_epoch=EPOCHS, lr=LEARNING_RATE, device=DEVICE):
    accuracies = []
    cnn = model.to(device)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)

    max_accuracy = 0

    for epoch in range(num_epoch):
        for i, (images, labels) in tqdm(enumerate(train_dl)):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
        accuracy = float(validate(cnn, val_dl))

        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("saving best model with accuracy: ", accuracy)
        print("Epoch: ", epoch + 1, "Accuracy: ", accuracy, "%")

    return best_model


CNN = train()

torch.save(CNN.state_dict(), "D12.pth")


def inference(path, model, device="cuda"):
    try:
        resp = requests.get(path, timeout=10)
        print("request sent")
    except:
        return False

    with torch.no_grad():
        image = np.array(Image.open(BytesIO(resp.content)))

        image = transform(image)

        image = image.unsqueeze(0)
        pred = model(image.to(device))
        return pred


path = "./data/test/"
pred = inference(path, model)
if torch.is_tensor(pred):
    pred_idx = np.argmax(pred)

    pred_label = "N" if pred_idx == 0 else "L"

    print(f"Predicted: {pred_label}, Prob: {pred[0][pred_idx] * 100}%")
else:
    print("can not get the url!!!")
