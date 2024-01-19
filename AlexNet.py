import torch
import time
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Set hyperparameters
epochs = 1
batch_size = 100
learning_rate = 0.001

# Record start time for performance measurement
start_time = time.time()

# MNIST dataset and data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define a Convolutional Neural Network (AlexNet-like architecture)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Convolutional layer 1: 1 input channel, 5 output channels, kernel size 5x5
        self.conv1 = nn.Conv2d(1, 5, 5)
        # Max pooling layer: kernel size 2x2, stride 1
        self.pool = nn.MaxPool2d(2, 1)
        # Convolutional layer 2: 5 input channels, 20 output channels, kernel size 5x5
        self.conv2 = nn.Conv2d(5, 20, 5)
        # Convolutional layer 3: 20 input channels, 20 output channels, kernel size 5x5
        self.conv3 = nn.Conv2d(20, 20, 5)
        # Convolutional layer 4: 20 input channels, 20 output channels, kernel size 5x5
        self.conv4 = nn.Conv2d(20, 20, 5)
        # Fully connected layer 1: input size 1620, output size 100
        self.fc1 = nn.Linear(1620, 100)
        # Fully connected layer 2: input size 100, output size 100
        self.fc2 = nn.Linear(100, 100)
        # Fully connected layer 3: input size 100, output size 10 (number of classes)
        self.fc3 = nn.Linear(100, 10)
        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv1 + Max Pooling + ReLU activation
        out = self.conv1(x)
        out = self.pool(out)
        out = self.relu(out)

        # Conv2 + Max Pooling + ReLU activation
        out = self.conv2(out)
        out = self.pool(out)
        out = self.relu(out)

        # Conv3 + Conv4 + Max Pooling + ReLU activation
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool(out)
        out = self.relu(out)

        # Reshape to fit into fully connected layer
        out = out.view(out.size(0), -1)

        # Fully connected layers with ReLU activation
        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        return out

# Create an instance of the ConvNet model
model = ConvNet()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Get the total number of batches for one epoch
n_total_steps = len(train_loader)
print(n_total_steps)

# Training loop
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        # Compute the loss
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print training information
        if (i+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    # Calculate and print the accuracy
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy on the test set: {acc:.2f}%')

# Record end time for performance measurement
end_time = time.time()
print(f'Time taken: {end_time - start_time:.2f} seconds')
