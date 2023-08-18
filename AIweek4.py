def method(x, y):
    print(x+y)

def main():
    # Your main code goes here
    print("This is the main method.")
    value = calculate_value(5, 7)
    print("Calculated value:", value)
    method(10, 4 )

def calculate_value(a, b):
    return a + b

if __name__ == "__main__":
    main()


