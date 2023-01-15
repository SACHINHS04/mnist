import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='.', train=True, download=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='.', train=False, download=True,
                              transform=transforms.ToTensor())

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the network and optimizer
model = Net()
optimizer = optim.Adam(model.parameters())

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Train the network
for epoch in range(10):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Print the current loss after each epoch
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Test the network on the test dataset
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print the final accuracy
    print(f'Accuracy: {100 * correct / total}')

# Save the model's state dictionary
torch.save(model.state_dict(), 'mnist_model.pth')
