import torch
import time
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Set hyperparameters
epochs = 1
batch_size = 50
learning_rate = 0.001
start_time = time.time()

# Load MNIST dataset
# Training dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
# Testing dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Create data loaders
# Training data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# Testing data loader
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define LeNet CNN model
class ConvNet(nn.Module):
    def __init__(self):
        # LeNet Convolution Neural Network architecture
        super(ConvNet, self).__init__()
        
        # First convolutional layer: input channels=1 (grayscale), output channels=5, kernel size=5x5
        self.conv1 = nn.Conv2d(1, 5, 5)
        
        # Max pooling layer: kernel size=2x2, stride=1
        self.pool = nn.MaxPool2d(2, 1)
        
        # Activation function (tanh)
        self.tanh = nn.Tanh()
        
        # Second convolutional layer: input channels=5, output channels=5, kernel size=5x5
        self.conv2 = nn.Conv2d(5, 5, 5)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 1)
        
        # Another tanh activation
        self.tanh = nn.Tanh()
        
        # Third convolutional layer: input channels=5, output channels=5, kernel size=5x5
        self.conv3 = nn.Conv2d(5, 5, 5)
        
        # Flatten the output before feeding it to fully connected layers
        self.flatten = nn.Flatten()
        
        # Fully connected layer 1: input features=980 (5x5x5x4), output features=100
        self.fc1 = nn.Linear(980, 100)
        
        # Tanh activation
        self.tanh = nn.Tanh()
        
        # Fully connected layer 2: input features=100, output features=10 (for 10 classes)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # Forward pass through the network
        
        # First convolutional layer, max pooling, and activation
        out = self.conv1(x)
        out = self.pool(out)
        out = self.tanh(out)
        
        # Second convolutional layer, max pooling, and activation
        out = self.conv2(out)
        out = self.pool(out)
        out = self.tanh(out)  
        
        # Third convolutional layer
        out = self.conv3(out)
        
        # Flatten the output
        out = self.flatten(out)
        
        # First fully connected layer and activation
        out = self.fc1(out)
        out = self.tanh(out)
        
        # Second fully connected layer (output layer)
        out = self.fc2(out)
        
        return out

# Instantiate model, loss function, and optimizer
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
steps = len(train_loader)

# Training loop
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss every 100 steps
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{steps}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
with torch.no_grad():
    correct = 0
    samples = 0
    for images, labels in test_loader:
        # Forward pass
        outputs = model(images)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        samples += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Print accuracy
    acc = 100.0 * correct / samples
    print(f'Accuracy: {acc:.2f}%')

# Calculate and print the total execution time
end_time = time.time()
print(f'Time required: {end_time - start_time:.2f} s')
