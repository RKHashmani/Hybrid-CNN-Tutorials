# @title Import the required modules
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define the "device". If GPU is available, device is set to use it, otherwise CPU will be used.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# @title Download the dataset
rand_transform = transforms.Compose([transforms.RandomChoice([
    transforms.Pad(3),
    transforms.RandomCrop(26),
    transforms.Pad(1),
    transforms.RandomCrop(27),
]), transforms.ToTensor()])
train_data = datasets.MNIST(root='../data', train=True,
                            transform=rand_transform, download=True)

test_data = datasets.MNIST(root='../data', train=False,
                           transform=rand_transform, download=True)

# @title Define the data loaders
batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# @title Define a CNN network
from MyFlexibleCNN import Net

# Create an instance
net = Net().to(device)

print(net)

# @title Define the loss function and the optimizer
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1.e-3)

# @title Train the model
num_epochs = 2
num_iters_per_epoch = 5000  # use only 5K iterations

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if i == num_iters_per_epoch:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = net(images)
        loss = loss_fun(output, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % (num_iters_per_epoch // 10) == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, num_iters_per_epoch, loss.item()))

# @title Run the trained model on the testing set
correct = 0
total = 0
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    out = net(images)
    _, predicted_labels = torch.max(out, 1)
    correct += (predicted_labels == labels).sum()
    total += labels.size(0)

print('Percent correct: %.3f %%' % ((100 * correct) / (total + 1)))