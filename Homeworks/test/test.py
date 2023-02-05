import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
import numpy as np
import os

# Load the FashionMNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root='data/',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)

test_dataset = torchvision.datasets.FashionMNIST(root='data/',
                                                  train=False,
                                                  transform=transforms.ToTensor(),
                                                  download=True)

# Create data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)

# Define the AlexNet model
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# # Initialize the model, loss function and optimizer
model = AlexNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# # Train the model
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/10], Step [{i+1}/600], Loss: {loss.item():.4f}')
    torch.save(model, './model_'+str(epoch)+'.pt')
    
# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_model = AlexNet()
test_model = torch.load('model1_9.pt', map_location=torch.device('cpu'))
test_model = test_model.to(device =device)

# The classes are: 

# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot


image = cv2.imread('./test_images/coat.jpg', cv2.IMREAD_GRAYSCALE)

# Load test images from local folder
test_images_folder = './test_images'
test_images = []
test_labels = []

test_images.append(image)
test_labels.append(4)

for image_name in os.listdir(test_images_folder):
    image = cv2.imread(os.path.join(test_images_folder, image_name), cv2.IMREAD_GRAYSCALE)
    label = image_name.split('_')[0] # assuming that the label is the first part of the file name, separated by '_'
    test_images.append(image)
    test_labels.append(label)


# Convert the images to tensors and normalize
test_images = torch.tensor(test_images, dtype=torch.float32) / 255
test_images = test_images.view(-1, 1, 28, 28)
test_labels = torch.tensor(test_labels)

# Evaluate the model on the test images
test_model.eval()
with torch.no_grad():
    outputs = test_model(test_images)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == test_labels).sum().item()
    accuracy = correct / len(test_labels)

print(f'Accuracy of the network on the local test images: {accuracy * 100}%')

