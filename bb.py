import amitUtils as utils
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
import matplotlib as plt


class Network(nn.Module):
    def __init__(
        self, input_size, output_size, learning_rate, momentum, hidden_layer_sizes, activation_type, batch_size, epochs
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs
        self.activation_type = activation_type

        print("Learning rate: ", learning_rate, "batch_size : ", batch_size, "Epochs : ", epochs)

        # create layers
        super(Network, self).__init__()

        layerlist = self.createLayerlist(input_size, output_size, hidden_layer_sizes, activation_type)
        self.model = nn.Sequential(layerlist)

    def createLayerlist(self, input_size, output_size, hidden_layer_sizes, activation_type):
        layers = OrderedDict()

        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        print(layer_sizes)

        layers = OrderedDict()
        for i, x in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.update({"linear" + str(i): nn.Linear(x[0], x[1])})
            layers.update({"sigmoid" + str(i): nn.Sigmoid()})
            # layers.update({"dropout" + str(i): nn.Dropout(0.1)})
            # layers.update({"relu" + str(i): nn.ReLU()})
        print(layers)

        return layers

    def forward(self, x):
        return self.model.forward(x)

    def trainer(self, dataloader):
        print("Data loader size: ", len(dataloader))  # dataloader)
        self.lossfn = nn.CrossEntropyLoss()
        losses = []
        epochs = []
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=0.05
        )
        for epoch in range(self.epochs):
            epoch_loss = 0
            steps = 0
            for data in iter(dataloader):
                # self.lossfn.zero_grad()  # zero the grads
                x = data[0]
                y = data[1]
                yhat = self.forward(x)
                loss = self.lossfn(yhat, y)
                self.lossfn.zero_grad()  # zero the grads
                loss.backward()  # accumulate gradients
                optimizer.step()  # Apply gradients to weights and biases
                steps += 1
                # if steps % 100 == 0:
                # print("Epoch: " + str(epoch) + "Steps: ", steps, "Loss " + str(loss.item()))
                epoch_loss += loss.item()
            steps = 0  # reset steps
            print("Epoch: " + str(epoch) + " Loss: " + str(epoch_loss / len((dataloader))))
            losses.append(epoch_loss / len(dataloader))
            epochs.append(epoch)
            self.validate()
        return epochs, losses
        # utils.plot(losses, epochs)
        # plt.plot(losses, epochs)
        # plt.show()

    def validate(self):  # Validate the results
        # dataloader = DataLoader(MnistDataset("test"), batch_size=100, shuffle=True)
        # with torch.no_grad():
        # for data in iter(dataloader):
        # x = data[0]
        # y = data[1]
        # yhat = self.forward(x)
        # print(yhat)
        with torch.no_grad():
            X, y = utils.load_fmnist_data("dev")
            yhat = self.forward(torch.from_numpy(X))
            validation_loss = self.lossfn(yhat, torch.from_numpy(y))
            print("Validation loss: " + str(validation_loss.item()))
            # print("Y Hat shape : " + str(yhat.shape), "Y shape : " + str(y.shape))

            # Calculate accuracy
            y = torch.from_numpy(y)
            correct = torch.argmax(y, dim=1) == torch.argmax(yhat, dim=1)
            accuracy = torch.mean(correct.float())

            print(f"Accuracy: {accuracy.item()}")
            # Calcualte Accuracy
            # accuracy = (yhat.argmax(dim=1) == torch.from_numpy(y)).float().mean()
            # print("Accuracy: " + str(accuracy.item()))


class MnistDataset(Dataset):
    def __init__(self, type):
        if type == "dev":
            self._X, self._labels = utils.load_fmnist_data("dev")
        elif type == "test":
            self._X, self._labels = utils.load_fmnist_data("test")
        elif type == "train":
            self._X, self._labels = utils.load_fmnist_data("train")
        else:
            raise ValueError("Type not supported")

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self._X[idx])
        y = torch.from_numpy(self._labels[idx])
        return x, y


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Network implementation : Assignment 3')
    parser.add_argument("-input_size", type=int, default=784)
    parser.add_argument("-output_size", type=int, default=10)
    parser.add_argument("-lr", type=float, default=0.01)
    # comma separated list of integers
    parser.add_argument("-sizes", type=list_int, default=[16, 32])
    # activation string - sigmoid, tanh
    parser.add_argument("-activation_type", type=str, default="sigmoid", choices=["sigmoid", "tanh"])
    # loss function - mse, ce
    parser.add_argument("-batch_size", type=int, default="1", help="1 or multiples of 5")
    # number of epochs`
    parser.add_argument("-epochs", type=int, default=1)
    args = parser.parse_args()
    print(args)
    print("Layers", args.sizes)
    net = Network(
        args.input_size,
        args.output_size,
        args.lr,
        args.sizes,
        args.momentum,
        args.activation_type,
        args.batch_size,
        args.epochs,
    )


def list_int(values):
    # return values.split(',').map(int)
    return [int(x) for x in values.split(',')]


if __name__ == "__main__":
    main()
