import torch


def tensors():
    t = torch.tensor([1,.2,2])
    print(t)
    s = torch.tensor([4,5,6])
    print(s)


def method(x, y):
    print(x+y)

def main(): # Your main code goes here
    print("This is the main method.")
    value = calculate_value(5, 7)
    print("Calculated value:", value)
    method(10, 4 )
    for i in range(10) : 
        print(i)

    tensors()

def calculate_value(a, b):
    return a + b

if __name__ == "__main__":
    main()
