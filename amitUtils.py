import numpy as np
import pandas as pd
import matplotlib as plt


# Load Fashion MNIST data from local file system and return a dataframe
def load_fmnist_data(type):
    if type == "dev":
        df = pd.read_csv("./fmnist/fashion-mnist_dev.csv")
    elif type == "test":
        df = pd.read_csv("./fmnist/fashion-mnist_test.csv")
    elif type == "train":
        df = pd.read_csv("./fmnist/fashion-mnist_train.csv")
    else:
        raise ValueError("Type not supported")

    # get only one row from the dataframe
    # df = df.sample(2)

    y = df["label"]
    x = df.drop("label", axis=1)

    x = x.to_numpy(dtype=np.float32)
    x = normalize(x)
    y = one_hot_encoding(y.to_numpy())
    return x, y


def one_hot_encoding(y):
    one_hot = np.zeros((len(y), 10))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # if x < 0:
    #     return np.exp(x) / (1 + np.exp(x))
    # else:
    #     return 1 / (1 + np.exp(-x))


# gradient of Relu
def relu_grad(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


# grad of sigmoid
# write code so that it does not cause runtime overflow
def sigmoid_grad(x):
    # Avoiding runtime overflow by using a more stable calculation
    sig = sigmoid(x)
    return sig * (1 - sig)


# Accuracy for classification
def accuracy(y_pred, y):
    y_pred = np.argmax(y_pred, axis=0)
    y = np.argmax(y, axis=0)
    return np.mean(y_pred == y)


def normalize(x):
    # return (x - x.min()) / (x.max() - x.min())
    return x / 255


def plot(x, y, type="line", title="MSE Plot"):
    import matplotlib.pyplot as mplt

    if type == "scatter":
        mplt.scatter(x, y)
    elif type == "line":
        mplt.plot(x, y)
    else:
        raise ValueError("Type not supported")
    mplt.title(title)
    mplt.show()


def plotImage(x):
    import matplotlib.pyplot as mplt

    mplt.imshow(x, cmap="gray")
