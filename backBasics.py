import numpy as np
import argparse
import amitUtils as utils
import logging as log

log.basicConfig(encoding='utf-8', level=log.DEBUG)


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
    ):
        self.momentum = momentum
        self.num_hidden = num_hidden
        self.layer_sizes = sizes
        self.activation_type = activation_type
        self.loss = loss
        self.opt = opt
        self.batch_size = batch_size
        self.anneal = anneal
        self.lr = lr
        self.epoch = 1
        self.input_size = input_size
        self.output_size = output_size
        layer_sizes = [self.input_size] + self.layer_sizes + [self.output_size]
        self.layer_sizes = layer_sizes
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            wt = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            # bias = np.random.randn(layer_sizes[i + 1], 1)
            bias = np.random.randn(layer_sizes[i + 1])
            l = Layer(wt, bias, self.activation_type)
            self.layers.append(l)
            log.debug("Number of layers created %s", len(self.layers))

        # For the last layer set the activation as softmax, if the loss function is ce - cross entropy
        if self.loss == "ce":
            self.layers[-1].activation_type = "softmax"
        # Construct weight matrices and bias for each layer starting backward

    def net_forward(self, x):
        # Now shape of x , (784, batch_size), x is now a collection of column vectors
        # for i in range(len(self.layers)):
        # x = self.layers[i].forward(x)
        for l in self.layers:
            x = l.forward(x)
            log.debug("Shape of activations %s", x.shape)
        return x

    def mse(self, output, label):
        # log.debug("Inside MSE : Output shape", output.shape, "label shape", label.shape)
        # return np.mean((label - output) ** 2)
        return np.sum((label - output) ** 2)

    def activation(self, x):
        # Sigmoid activation
        return utils.sigmoid(x)

    # We will write three methods, develop, train and test
    def act(self, epochs, dataset):
        # x, y = utils.load_fmnist_data("dev")
        X, Y = utils.load_fmnist_data(dataset)
        log.debug("Pre Batch shape : %s %s", X.shape, Y.shape)
        # X Shape (N, 784) Y Shape (N, 10)
        # N = 100,for development, 100 rows and 784 columns, X is collection of row vectors of 28*28 = 784
        # Break the data into batches of batch_size
        batch_size = self.batch_size
        num_batches = len(X) // batch_size
        mse = []
        epoch_list = []
        for j in range(epochs):
            squared_error = 0
            for i in range(num_batches):
                batch_x = X[i * batch_size : (i + 1) * batch_size]
                batch_y = Y[i * batch_size : (i + 1) * batch_size]
                log.debug("Batch shape %s %s", batch_x.shape, batch_y.shape)
                # Select rows equal to the batch size from x and y, so batch shape (batch_size, 784), batch is still row vectors
                # Process the batch here, and do a transpose
                for l in self.layers:  # Clear the grad for each layer before new batch
                    l.clear_grad()
                for x, y in zip(batch_x, batch_y):
                    output = self.net_forward(x.T)
                    self.backward(x.T, output, y.T)  # Accumulate the gradients
                for l in self.layers:
                    l.backward(self.lr, batch_size)

            # log.debug output and label shapes
            log.debug("Output shape: %s", output.shape)
            log.debug("Label shape: %s", Y.shape)

            # log.debug("Squared Error :", squared_error / batch_size, "Epoch :", j)
            log.info("Squared Error : %s", squared_error, "Epoch : %s", j)
            mse.append(squared_error)
            epoch_list.append(j)
            log.debug(("Last Layer Activations %s", self.layers[-1].activations))
        utils.plot(mse, epoch_list)

    def backward(self, input, output, label):
        # Now , output size is (10,batch_size)
        # Now , label size is (10,batch_size)

        # Equation 1 : Output error at the last layer , grad of loss function with respect to activation multiplied by derivative of activation function
        # Loss Grad is just a element wise difference so it is (10,batch_size)
        # Activation Grad is (10,batch_size)
        # Below is element wise multiplication so output error is (10,batch_size)
        # output_error = self.getLossgrad(output, label) * self.getActivationGrad(output)
        output_error = self.getLossgrad(output, label) * self.getActivationGrad(self.layers[-1].z)

        # output_error = np.mean(output_error, axis=1, keepdims=True)
        log.debug("Output error shape %s", output_error.shape)
        # This is the error of the last layer, set it in the last layer
        self.layers[-1].delta = output_error

        for i in reversed(range(len(self.layers))):
            log.debug("Layers weights %s", self.layers[i].weights.shape)

            # Equation 2 : Error at each of the hidden layers
            # check if current layer is the last layer
            if i == len(self.layers) - 1:
                pass
            else:
                self.layers[i].delta = self.layers[i + 1].weights.T.dot(
                    self.layers[i + 1].delta
                ) * self.getActivationGrad(self.layers[i].z)

                log.debug(
                    "All shapes in order of the appearance %s %s %s",
                    self.layers[i + 1].weights.T.shape,
                    self.layers[i + 1].delta.shape,
                    self.getActivationGrad(self.layers[i].z).shape,
                )
            # weights shape ((30,10)) delta shape ((10,100)) activation grad shape ((30,100))

            # self.layers[i].bias_grad = self.layers[i].delta
            bias_grad = self.layers[i].delta

            # self.layers[i].bias_grad = np.mean(self.layers[i].delta, axis=1, keepdims=True)

            # Equation 3 : Gradient of weights for each layer, dot product with bias_grad because it is equal to the del error in this layer
            # dot product of previous layer activations and current layer del error
            weights_grad = None
            if i == 0:
                # self.layers[i].weights_grad = np.dot(self.layers[i].delta, input.T)
                # self.layers[i].weights_grad = np.outer(self.layers[i].delta, input.T)
                weights_grad = np.outer(self.layers[i].delta, input.T)
                log.debug("Inside weights grad first layer %s %s", self.layers[i].delta.shape, input.T.shape)
            else:
                # Do mean of activations to reduce the dimensions
                # log.debug("Inside weights grad ", self.layers[i].delta.shape, self.layers[i - 1].activations.T.shape)
                # self.layers[i - 1].activations = np.mean(self.layers[i - 1].activations, axis=1, keepdims=True)
                # log.debug("Inside weights grad ", self.layers[i].delta.shape, self.layers[i - 1].activations.T.shape)

                # self.layers[i].weights_grad = np.dot(self.layers[i].delta, self.layers[i - 1].activations.T)
                # self.layers[i].weights_grad = np.outer(self.layers[i].delta, self.layers[i - 1].activations.T)
                weights_grad = np.outer(self.layers[i].delta, self.layers[i - 1].activations.T)

            self.layers[i].accumulate_grad(bias_grad, weights_grad)

            log.debug("Weights grad shape %s", weights_grad.shape)
            log.debug("Previous Layer's Activations shape %s", self.layers[i - 1].activations.T.shape)

            # Update the weights and biases for each layer

            # self.layers[i].backward(self.lr)

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
        log.debug("Shape of weights %s", self.weights.shape)
        log.debug("Shape of bias %s", self.biases.shape)
        # provide a seed so that same random numbers are generated on each run
        # Store the z value and activation
        self.z = None
        self.activations = np.array([self.biases.shape])
        # Bias grad at each neuron as per the book
        self.bias_grad = np.zeros((self.biases.shape))
        # Weights grad at each neuron
        self.weights_grad = np.zeros((self.weights.shape))
        # Del error at each neuron, shape is same as bias
        self.delta = np.array([self.biases.shape])

    def forward(self, x):
        log.debug("Shapes in layer forward %s %s %s", self.weights.shape, x.shape, self.biases.shape)
        # X is column vector, for 1st layer (layer_size,784) dot(784, batch_size) = (layer_size, batch_size)+ (layer_size, 1) - last term will be broadcasted
        # so output will be (layer_size, batch_size), which is 16 columns vectors of size 128 for each neuron of the layer
        # its size will not change due to activation function which will be broadcasted element wise
        # second and next layers will be (weights of (current_layer_size, layer_size))(layer_size, batch_size)= (current_layer_size, batch_size) + (current_layer_size, 1) - last term will be broadcasted
        # so output will be (current_layer_size, batch_size) which is 16 columns vectors of size current_layer_size which will be 128,16,32, and so on.
        # In the end we will then get (10, batch_size) where 10 = output_size = size of the output layer
        # Means 16 column vectors of size 10
        # x = np.expand_dims(x, axis=1)
        # log.debug("Shape after expand %s", x.shape)
        self.z = np.dot(self.weights, x) + self.biases
        log.debug("Shape after matmul %s", self.z.shape)
        if self.activation_type == "sigmoid":
            self.activations = utils.sigmoid(self.z)
        elif self.activation_type == "softmax":
            self.activations = utils.softmax(self.z)
        else:
            # throw value exception if the activation type is not supported
            raise ValueError("Activation type not supported")
        return self.activations

    def backward(self, lr, batch_size):
        # Update the weights and biases for the batch size
        # mean = np.mean(x, axis=1, keepdims=True)
        self.weights = self.weights - lr * self.weights_grad
        self.biases = self.biases - lr * self.bias_grad
        # Mean to reduce dimensions
        # self.biases = np.mean(self.biases, axis=1, keepdims=True)
        log.debug("Weights, Bias shape in layer backward %s %s", self.weights.shape, self.biases.shape)

    def accumulate_grad(self, bias_grad, weights_grad):
        # delta = delta.T
        self.bias_grad += bias_grad
        self.weights_grad += weights_grad

    def clear_grad(self):
        self.bias_grad = np.empty((self.biases.shape))
        self.weights_grad = np.empty((self.weights.shape))


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
    log.debug(args)
    log.debug("Layers %s", args.sizes)
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
    )

    net.act(args.epochs, "train")

    # Accuracy
    x, y = utils.load_fmnist_data("test")  # utils.load_data("dev")
    output = net.net_forward(x.T)
    # output = np.mean(output, axis=0, keepdims=True)
    log.debug("Output shape %s", output.shape)
    log.debug("Y shape %s", y.T.shape)
    accuracy = np.mean(np.argmax(output.T, axis=1) == np.argmax(y, axis=1))
    # log.debug(np.argmax(output.T, axis=1))
    # log.debug(np.argmax(y, axis=1))
    log.info("Accuracy", accuracy)


def list_int(values):
    # return values.split(',').map(int)
    return [int(x) for x in values.split(',')]


if __name__ == "__main__":
    main()
