import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch

import torch.nn as nn
from torch import optim
import torch.nn.functional as F

if __name__ == '__main__':

    def imshow(img, text=None):
        npimg = img.numpy()
        plt.axis("off")
        if text:
            plt.text(75, 8, text, style='italic', fontweight='bold',
                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    def show_plot(iteration, loss):
        plt.plot(iteration, loss)
        plt.show()


    class SiameseNetworkDataset(Dataset):
        def __init__(self, imageFolderDataset, transform=None):
            self.imageFolderDataset = imageFolderDataset
            self.transform = transform

        def __len__(self):
            return len(self.imageFolderDataset.imgs)

        def __getitem__(self, index):
            img0_tuple = random.choice(self.imageFolderDataset.imgs)

            # We need to approximately 50% of images to be in the same class
            should_get_same_class = random.randint(0, 1)
            if should_get_same_class:
                while True:
                    # Look untill the same class image is found
                    img1_tuple = random.choice(self.imageFolderDataset.imgs)
                    if img0_tuple[1] == img1_tuple[1]:
                        break
            else:

                while True:
                    # Look untill a different class image is found
                    img1_tuple = random.choice(self.imageFolderDataset.imgs)
                    if img0_tuple[1] != img1_tuple[1]:
                        break

            img0 = Image.open(img0_tuple[0])
            img1 = Image.open(img1_tuple[0])

            img0 = img0.convert("L")
            img1 = img1.convert("L")

            if self.transform is not None:
                img0 = self.transform(img0)
                img1 = self.transform(img1)

            return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))


    # Load the training dataset
    folder_dataset = datasets.ImageFolder(root="./data/training/")
    # Resize the images and transform to tensors
    transformation = transforms.Compose([transforms.Resize((100, 100)),
                                         transforms.ToTensor()
                                         ])

    # Initialize the network
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transformation)
    # Create a simple dataloader just for simple visualization
    vis_dataloader = DataLoader(siamese_dataset,
                                shuffle=True,
                                num_workers=0,
                                batch_size=12)

    # Extract one batch
    example_batch = next(iter(vis_dataloader))

    # Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
    # If the label is 1, it means that it is not the same person, label is 0, same person in both images
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)

    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy().reshape(-1))


    # create the Siamese Neural Network
    class SiameseNetwork(nn.Module):

        def __init__(self):
            super(SiameseNetwork, self).__init__()

            # Setting up the Sequential of CNN Layers
            # noinspection PyTypeChecker
            self.cnn1 = nn.Sequential(
                nn.Conv2d(1, 96, 11, stride=4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2),

                nn.Conv2d(96, 256, 5, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(256, 384, 3, stride=1),
                nn.ReLU(inplace=True)
            )

            # Setting up the Fully Connected Layers
            self.fc1 = nn.Sequential(
                nn.Linear(384, 1024),
                nn.ReLU(inplace=True),

                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),

                nn.Linear(256, 2)
            )

        def forward_once(self, x):
            # This function will be called for both images
            # It's output is used to determine the similiarity
            output = self.cnn1(x)
            output = output.view(output.size()[0], -1)
            output = self.fc1(output)
            return output

        def forward(self, input1, input2):
            # In this function we pass in both images and obtain both vectors
            # which are returned
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)

            return output1, output2


    # Define the Contrastive Loss Function
    class ContrastiveLoss(torch.nn.Module):
        def __init__(self, margin=2.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

        def forward(self, output1, output2, label):
            # Calculate the euclidian distance and calculate the contrastive loss
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

            loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

            return loss_contrastive


    # Load the training dataset
    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=0,
                                  batch_size=12)
    net = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    counter = []
    loss_history = []
    iteration_number = 0

    # Iterate throught the epochs
    for epoch in range(100):
        # Iterate over batches
        for i, (img0, img1, label) in enumerate(train_dataloader, 0):

            # Send the images and labels to CUDA
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs
            output1, output2 = net(img0, img1)

            # Pass the outputs of the networks and label into the loss function
            loss_contrastive = criterion(output1, output2, label)

            # Calculate the backpropagation
            loss_contrastive.backward()

            # Optimize
            optimizer.step()

            # Every 10 batches print out the loss
            if i % 10 == 0:
                print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
                iteration_number += 10

                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
               [r'$0$', r'$5$', r'$10$', r'$15$', r'$20$', r'$25$', r'$30$', r'$35$', r'$40$', r'$45$', r'$50$'])
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    show_plot(counter, loss_history)

    # Locate the test dataset and load it into the SiameseNetworkDataset
    folder_dataset_test = datasets.ImageFolder(root="./data/testing/")
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transformation)
    test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=12, shuffle=True)

    # Grab one image that we are going to test
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)

    for i in range(10):
        # Iterate over 10 images and test them with the first image (x0)
        _, x1, label2 = next(dataiter)

        # Concatenate the two images together
        concatenated = torch.cat((x0, x1), 0)

        output1, output2 = net(x0.cuda(), x1.cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
