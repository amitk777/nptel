import bb
import amitUtils as utils
from torch.utils.data import DataLoader

batch_size = 20
epochs = 20
hidden_layers = [200, 200]
learning_rate = 0.00001
momentum = 0.9
activation_type = "sigmoid"

dataset = bb.MnistDataset("train")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# print(next(iter(dataloader))[0].shape)

net = bb.Network(
    input_size=28 * 28,
    output_size=10,
    learning_rate=learning_rate,
    momentum=momentum,
    hidden_layer_sizes=hidden_layers,
    activation_type=activation_type,
    batch_size=batch_size,
    epochs=epochs,
)
epochs, losses = net.trainer(dataloader)
print(epochs, losses)
# utils.plot(epochs, losses)
# print the weights type of hte first layer
# layer = torch.nn.Linear(28 * 28, 100)
# print(laer.weight.dtype)
