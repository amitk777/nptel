# Define a CNN model
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import pandas as pd
import torchmetrics


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First convolutional layer: 1 input channel, 32 output channels, 3x3 kernel
        # self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv1 = nn.Conv2d(1, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)  # Max-pooling layer with 2x2 window

        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(64, 128, 3)

        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 256, 3)

        # First fully connected (linear) layer: 64 * 5 * 5 input features, 128 output features
        self.fc1 = nn.Linear(256, 1024)
        # Second fully connected (linear) layer: 128 input features, 10 output features (classes)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply first convolution, ReLU, and max-pooling
        # Size is now W_out = (W_in - K + 2*P) / S + 1
        # Size is now W_out = (28 - 3 + 2*0) / 1 + 1 = 26
        # Max pool 2*2 => Size will be 13*13
        x = self.pool(torch.relu(self.conv2(x)))  # Apply second convolution, ReLU, and max-pooling
        # Size is now W_out = (13 - 3 + 2*0) / 1 + 1 = 11
        # Max pool 2*2 => Size will be 5*5
        x = torch.relu(self.conv3(x))  # Apply third convolution, ReLU
        # Size is now W_out = (5 - 3 + 2*0) / 1 + 1 = 3
        # x = self.pool(torch.relu(self.conv4(x)))  # Apply fourth convolution, ReLU, and max-pooling
        x = torch.relu(self.conv4(x))  # Apply fourth convolution, ReLU, and max-pooling
        # Size is now W_out = (3 - 3 + 2*0) / 1 + 1 = 1
        # What to be done for pooling
        x = x.view(-1, 256 * 1 * 1)  # Reshape for fully connected layers
        # x = nn.BatchNorm2d(64 * 5 * 5)
        x = torch.relu(self.fc1(x))  # Apply first fully connected layer and ReLU
        x = self.fc2(x)  # Apply second fully connected layer
        # Apply softmax
        x = nn.Softmax(dim=1)(x)  # Apply softmax

        return x


class CustomCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file.
            transform (callable, optional): Optional transform to be applied to the data.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract label from the first column
        label = torch.tensor(self.data.iloc[idx, 0], dtype=torch.long)
        label = one_hot_encoding(label)

        # label = one_hot_encoding(label)

        # features are reshaped for image dimensions 28*28 , which is required for filters and unsqueeze adds a dimension for channel 1 which is for grayscale
        features = torch.tensor(self.data.iloc[idx, 1:], dtype=torch.float32).reshape(28, 28).unsqueeze(0)

        # sample = {"features": features, "label": label}

        # if self.transform:
        # sample = self.transform(sample)

        return features, label


def one_hot_encoding(label):
    one_hot = torch.zeros(10)
    one_hot[label] = 1
    # print("Size recived by one hot method", label.size)
    return one_hot


# Load and preprocess the MNIST dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


# transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
#
def trainNetwork(epochs=5):
    trainset = CustomCSVDataset("./data/fashion-mnist_train.csv", transform=None)
    #
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    # #
    # # Create an instance of the CNN model
    net = Net()
    #
    # # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #
    # # Train the model
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for _, data in enumerate(trainloader, 0):
            inputs, labels = data
            # print("Output from trainloder, for inputs", inputs.shape)
            optimizer.zero_grad()
            outputs = net(inputs)
            # print("Output Shape", outputs.shape)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")
    #
    print("Finished Training")
    # Save the trained model
    torch.save(net.state_dict(), "mnist_cnn.pth")


# Instantiate the custom dataset
# csv_file = 'your_dataset.csv'
# transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
# As per the NPTEL , no need to do transformation
# custom_dataset = CustomCSVDataset("./data/fashion-mnist_test.csv", transform=None)
# print("Lenght of custom dataset", len(custom_dataset))
def testConvolution():
    #  Access an item in the dataset
    custom_dataset = CustomCSVDataset("./data/fashion-mnist_test.csv", transform=None)
    # sample = custom_dataset[0]
    sample = custom_dataset
    print(sample[0])
    # #
    # Access the features and label
    features, label = sample[0]
    print("Types: ", type(features), type(label))
    print("Features : ", features.shape)
    # #
    print("Label : ", label.shape)
    print("Features : ", features.shape)
    print(features.type())
    print(features.shape)
    conv1 = nn.Conv2d(1, 64, 3)
    pool = nn.MaxPool2d(2, 2)  # Max-pooling layer with 2x2 window
    x = features.reshape(28, 28).unsqueeze(0)
    # features = features.unsqueeze(0)
    print("Shape after unsqueeze", x.shape)
    x = pool(torch.relu(conv1(x)))  # Apply first convolution, ReLU, and max-pooling

    print("Shape after 1st convolution", x.shape)


def validateNetwork():
    model = Net()
    model.load_state_dict(torch.load("mnist_cnn.pth"))
    model.eval()
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # testConvolution()# Initialize variables to keep track of validation loss and accuracy
    total_loss = 0.0

    correct_predictions = 0
    total_predictions = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    validation_set = CustomCSVDataset("./data/fashion-mnist_test.csv", transform=None)
    #
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=True)
    # #
    validation_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    # Disable gradient computation for validation
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            # accuracy = validation_accuracy(outputs, labels)
            # print("Accuracy on batch ", accuracy)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Accumulate loss
            total_loss += loss.item()

    # Calculate validation loss and accuracy
    validation_loss = total_loss / len(validation_loader)
    validation_accuracy = correct_predictions / total_predictions

    print(f"Validation Loss: {validation_loss:.4f}")
    print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")


# validateNetwork()
# Main function which accepts a String arguement and calls appropriate function , testConvolution(), validateNetwork() or trainnetwork()


def __main__(arg):
    if arg == "test":
        testConvolution()
    elif arg == "validate":
        print("Validating")
        validateNetwork()
    elif arg == "train":
        trainNetwork()


# Check if being called as main program

if __name__ == "__main__":
    __main__(sys.argv[1])

# # Print the features which is a image via matplotlib
# import matplotlib.pyplot as plt
#
# plt.imshow(features.reshape(28, 28), cmap='gray')
"""
Document of the work done
Target Accuracy : 97%
5 Epochs of training = Accuracy is 87.8%
10 Epochs of training = Accuracy is 89.52%

"""
# plt.show(
