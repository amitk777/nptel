import numpy as np
import argparse
import amitUtils as utils


class Network:
    """takes input the sizes of various layers, sizes is an array of the sizes of various layers"""

    def __init__(
        self,
        input_size,
        output_size,
        lr,
        momentum,
        num_hidden,
        sizes,
        activation_type,
        loss,
        opt,
        batch_size,
        anneal,
        save_dir,
        expt_dir,
        train,
        test,
    ):
        self.momentum = momentum
        self.num_hidden = num_hidden
        self.layer_sizes = sizes
        self.activation_type = activation_type
        self.loss = loss
        self.opt = opt
        self.batch_size = batch_size
        self.anneal = anneal
        self.save_dir = save_dir
        self.expt_dir = expt_dir
        self.train = train
        self.test = test
        self.lr = lr
        self.epoch = 1
        self.input_size = input_size
        self.output_size = output_size
        layer_sizes = [self.input_size] + self.layer_sizes + [self.output_size]
        self.layer_sizes = layer_sizes
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            np.random.seed(1)
            wt = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            bias = np.random.randn(layer_sizes[i + 1], 1)
            l = Layer(wt, bias, self.activation_type)
            self.layers.append(l)
            print("Number of layers created", len(self.layers))
        # Construct weight matrices and bias for each layer starting backward

    def forward(self, x):
        # Now shape of x , (784, batch_size), x is now a collection of column vectors
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
            print("Shape of activations", x.shape)
        return x

    def mse(self, output, label):
        # print("Inside MSE : Output shape", output.shape, "label shape", label.shape)
        # return np.mean((label - output) ** 2)
        return np.sum((label - output) ** 2)

    def activation(self, x):
        # Sigmoid activation
        return 1 / (1 + np.exp(-x))

    # We will write three methods, develop, train and test
    def act(self, epochs):
        x, y = utils.load_fmnist_data("dev")
        print("Pre Batch shape", x.shape, y.shape)
        # X Shape (N, 784) Y Shape (N, 10)
        # N = 100,for development, 100 rows and 784 columns, X is collection of row vectors of 28*28 = 784
        # Break the data into batches of batch_size
        batch_size = self.batch_size
        num_batches = len(x) // batch_size
        remaining_samples = len(x) % batch_size
        for j in range(epochs):
            squared_error = 0
            for i in range(num_batches):
                batch_x = x[i * batch_size : (i + 1) * batch_size]
                batch_y = y[i * batch_size : (i + 1) * batch_size]
                print("Batch shape", batch_x.shape, batch_y.shape)
                # Select rows equal to the batch size from x and y, so batch shape (batch_size, 784), batch is still row vectors
                # Process the batch here, and do a transpose
                output = self.forward(batch_x.T)
                self.backward(batch_x.T, output, batch_y.T)
                # print("Output shape,net output shape before collecting :", output.shape, net_output.shape)
                squared_error = squared_error + self.mse(output, batch_y.T)

            if remaining_samples > 0:
                batch_x = x[num_batches * batch_size :]
                batch_y = y[num_batches * batch_size :]
                print("Batch shape", batch_x.shape, batch_y.shape)
                # Process the remaining samples here
                output = self.forward(batch_x.T)
                self.backward(batch_x.T, output, batch_y.T)
                squared_error = squared_error + self.mse(output, batch_y.T)
            # Print output and label shapes
            print("Output shape:", output.shape)
            print("Label shape:", y.shape)

            # print("Squared Error :", squared_error / batch_size, "Epoch :", j)
            print("Squared Error :", squared_error, "Epoch :", j)

    def backward(self, input, output, label):
        # Now , output size is (10,batch_size)
        # Now , label size is (10,batch_size)

        # Equation 1 : Output error at the last layer , grad of loss function with respect to activation multiplied by derivative of activation function
        # Loss Grad is just a element wise difference so it is (10,batch_size)
        # Activation Grad is (10,batch_size)
        # Below is element wise multiplication so output error is (10,batch_size)
        output_error = self.getLossgrad(output, label) * self.getActivationGrad(output)
        # Take average output error
        # output_error = np.mean(output_error, axis=1, keepdims=True)
        print("Output error shape", output_error.shape)
        # This is the error of the last layer, set it in the last layer
        self.layers[-1].delta = output_error

        for i in reversed(range(len(self.layers))):
            print("Layers weights", self.layers[i].weights.shape)

            # Equation 2 : Error at each of the hidden layers
            # check if current layer is the last layer
            if i == len(self.layers) - 1:
                pass
            else:
                self.layers[i].delta = self.layers[i + 1].weights.T.dot(
                    self.layers[i + 1].delta
                ) * self.getActivationGrad(self.layers[i].z)

                print(
                    "All shapes in order of the appearance",
                    self.layers[i + 1].weights.T.shape,
                    self.layers[i + 1].delta.shape,
                    self.getActivationGrad(self.layers[i].z).shape,
                )

            self.layers[i].bias_grad = self.layers[i].delta

            # Equation 3 : Gradient of weights for each layer, dot product with bias_grad because it is equal to the del error in this layer
            # dot product of previous layer activations and current layer del error
            if i == 0:
                self.layers[i].weights_grad = np.dot(self.layers[i].delta, input.T)
                print("Inside weights grad first layer ", self.layers[i].delta.shape, input.T.shape)
            else:
                self.layers[i].weights_grad = np.dot(self.layers[i].delta, self.layers[i - 1].activations.T)
                print("Inside weights grad ", self.layers[i].delta.shape, self.layers[i - 1].activations.T.shape)

            print("Weights grad", self.layers[i].weights_grad.shape)
            print("Previous Layer's Activations shape", self.layers[i - 1].activations.T.shape)

            # Update the weights and biases for each layer

            self.layers[i].backward(self.lr)

    def getLossgrad(self, activation, label):
        lossGrad = np.array(activation.shape)
        if self.loss == "mse":
            lossGrad = activation - label
            return lossGrad
        else:
            raise ValueError("Loss function not supported")

    def getActivationGrad(self, activation):
        if self.activation_type == "sigmoid":
            # return activation * (1 - activation)
            return utils.sigmoid_grad(activation)
        else:
            raise ValueError("Activation type not supported")


class Layer:
    # A layer is characterzied by weights, bias , activations and activation type
    def __init__(self, weights, biases, activation_type):
        # Transpose the weights so that we get the right size
        self.weights = weights.T
        self.biases = biases
        self.activation_type = activation_type
        print("Shape of weights", self.weights.shape)
        print("Shape of bias", self.biases.shape)
        # provide a seed so that same random numbers are generated on each run
        # Store the z value and activation
        self.z = None
        self.activations = np.array([self.biases.shape], dtype=np.float64)
        # Bias grad at each neuron as per the book
        self.bias_grad = np.array([self.biases.shape])
        # Weights grad at each neuron
        self.weights_grad = np.array([self.weights.shape])
        # Del error at each neuron, shape is same as bias
        self.delta = np.array([self.biases.shape])

    def forward(self, x):
        print("Shapes in layer forward", self.weights.shape, x.shape, self.biases.shape)
        # X is column vector, for 1st layer (layer_size,784) dot(784, batch_size) = (layer_size, batch_size)+ (layer_size, 1) - last term will be broadcasted
        # so output will be (layer_size, batch_size), which is 16 columns vectors of size 128 for each neuron of the layer
        # its size will not change due to activation function which will be broadcasted element wise
        # second and next layers will be (weights of (current_layer_size, layer_size))(layer_size, batch_size)= (current_layer_size, batch_size) + (current_layer_size, 1) - last term will be broadcasted
        # so output will be (current_layer_size, batch_size) which is 16 columns vectors of size current_layer_size which will be 128,16,32, and so on.
        # In the end we will then get (10, batch_size) where 10 = output_size = size of the output layer
        # Means 16 column vectors of size 10

        self.z = np.dot(self.weights, x) + self.biases
        if self.activation_type == "sigmoid":
            self.activations = self.sigmoid(self.z)
            return self.activations
        else:
            # throw value exception if the activation type is not supported
            raise ValueError("Activation type not supported")

    def backward(self, lr):
        # Update the weights and biases for the batch size
        # mean = np.mean(x, axis=1, keepdims=True)
        self.weights = self.weights - lr * self.weights_grad
        self.biases = self.biases - lr * self.bias_grad
        print("Weights, Bias shape in layer backward", self.weights.shape, self.biases.shape)

    # activation - sigmoid function, since our outputs are between 0 and 1
    def sigmoid(self, x):
        # x = np.clip(x, -500, 500)  # Clipping values to prevent overflow
        return 1 / (1 + np.exp(-x))

    # as per the book
    def error(self, x, y):
        return np.mean((y - x) ** 2)


# Use argparse module to parse the arguments in the main method.


def main():
    import pathlib

    parser = argparse.ArgumentParser(description='Network implementation : Assignment 3')
    parser.add_argument("-input_size", type=int, default=10)
    parser.add_argument("-output_size", type=int, default=10)
    parser.add_argument("-lr", type=float, default=0.01)
    parser.add_argument("-momentum", type=float, default=0.1)
    parser.add_argument("-num_hidden", type=int, default=1)
    # comma separated list of integers
    parser.add_argument("-sizes", type=list_int, default=[16, 32])
    # activation string - sigmoid, tanh
    parser.add_argument("-activation_type", type=str, default="sigmoid", choices=["sigmoid", "tanh"])
    # loss function - mse, ce
    parser.add_argument("-loss", type=str, default="mse", choices=["mse", "ce"])
    # optimization algorithm, gd, momentum,nag ,adam
    parser.add_argument("-opt", type=str, default="adam", choices=["gd", "momentum", "nag", "adam"])
    parser.add_argument("-batch_size", type=int, default="1", help="1 or multiples of 5")
    parser.add_argument(
        "-anneal",
        type=bool,
        default="False",
        help="learning rate is halved if validation loss decreases and then restart the epoch",
    )
    # path where the pickled model will be saved
    # parser.add_argument("-save_dir", type=argparse.FileType('w'), default="/home/amit/projects/nptel")
    parser.add_argument("-save_dir", type=pathlib.Path, default="/home/amit/projects/nptel")
    # path where the log file be saved
    parser.add_argument("-expt_dir", type=argparse.FileType('w'), default="/home/amit/projects/nptel/expt.log")
    # path to train dataset
    parser.add_argument("-train", type=pathlib.Path, default="/home/amit/projects/nptel")
    # path to test dataset
    parser.add_argument("-test", type=pathlib.Path, default="/home/amit/projects/nptel")
    # number of epochs`
    parser.add_argument("-epochs", type=int, default=1)
    args = parser.parse_args()
    print(args)
    print("Layers", args.sizes)
    net = Network(
        args.input_size,
        args.output_size,
        args.lr,
        args.momentum,
        args.num_hidden,
        args.sizes,
        args.activation_type,
        args.loss,
        args.opt,
        args.batch_size,
        args.anneal,
        args.save_dir,
        args.expt_dir,
        args.train,
        args.test,
    )

    net.act(args.epochs)


def list_int(values):
    # return values.split(',').map(int)
    return [int(x) for x in values.split(',')]


if __name__ == "__main__":
    main()
