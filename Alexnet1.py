import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
epochs = 1
batch_size = 100
learning_rate = 0.001
start_time=time.time()
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data',  train=True,   transform=transforms.ToTensor(),    download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',  train=False,   transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  batch_size=batch_size,   shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size,  shuffle=False)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.pool = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(5, 20, 5)
        self.conv3 = nn.Conv2d(20, 20, 5)
        self.conv4 = nn.Conv2d(20, 20, 5)
        self.fc1 = nn.Linear(1620, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100,10)
        self.relu = nn.ReLU()

    def forward(self, x):
        out=self.conv1(x)
        out=self.pool(out)
        out=self.relu(out)
        
        out=self.conv2(out)
        out=self.pool(out)
        out=self.relu(out)
        
        out=self.conv3(out)
        out=self.conv4(out)
        out=self.pool(out)
        out=self.relu(out)
        
        out = out.view(out.size(0), -1)
        out=self.fc1(out)
        out = self.relu(out)
        
        
        out=self.fc2(out)
        out = self.relu(out)
        
        out=self.fc3(out)
        return out

model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
n_total_steps = len(train_loader)
print(n_total_steps)

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 20 == 0:
            print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(acc)
end_time=time.time()
print(f'time : {end_time-start_time}')