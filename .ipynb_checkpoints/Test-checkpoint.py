import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate some synthetic data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


# Define a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature, one output

    def forward(self, x):
        return self.linear(x)


# Instantiate the model
model = LinearRegression()

# Define the loss function (mean squared error)
criterion = nn.MSELoss()

# Define the optimizer (Adagrad)
optimizer = optim.Adagrad(model.parameters(), lr=0.1)


# Training loop

def main():
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Print the trained parameters
    with torch.no_grad():
        print("Trained parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)


if __name__ == "__main__":
    main()