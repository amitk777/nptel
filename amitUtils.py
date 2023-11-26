import numpy as np
import pandas as pd


def testFunction(x):
    print("Printing  from module", x)


# Load Fashion MNIST data from local file system and return a dataframe
def load_fmnist_data(type):
    if type == "dev":
        df = pd.read_csv("./fmnist/fashion-mnist_train.csv")
    elif type == "test":
        df = pd.read_csv("./fmnist/fashion-mnist_test.csv")
    elif type == "train":
        df = pd.read_csv("./fmnist/fashion-mnist_train.csv")
    else:
        raise ValueError("Type not supported")

    y = df["label"]
    x = df.drop("label", axis=1)
    return x, y
