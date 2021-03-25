import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

from MyMLP import MLP

# Define the "device". If GPU is available, device is set to use it, otherwise CPU will be used.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Download the dataset
train_data = datasets.MNIST(root='./data', train=True,
                            transform=transforms.ToTensor(), download=True)

test_data = datasets.MNIST(root='./data', train=False,
                           transform=transforms.ToTensor())

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=False)

# Create a MLP network instance

input_size = 28 * 28 # Size of MNIST
num_hidden_neurons = 64
num_classes = 10

net = MLP(input_size, num_hidden_neurons, num_classes)
net.to(device)
print(net)

# Define the loss function and the optimizer
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( net.parameters(), lr=1.e-3)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28 * 28).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = net(images)
        loss = loss_fun(output, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % batch_size == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_data) // batch_size, loss.item()))


# Run the trained model on the testing set
correct = 0
total = 0
for images, labels in test_loader:
    images = images.view(-1, 28 * 28).to(device)
    labels = labels.to(device)

    out = net(images)
    _, predicted_labels = torch.max(out, 1)
    correct += (predicted_labels == labels).sum()
    total += labels.size(0)

print('Percent correct: %.3f %%' % ((100 * correct) / (total + 1)))