import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df  = pd.read_csv("tilt_data.csv")

x = df['session_seconds'].values.astype(float)
y = df['session_losses'].values.astype(float)

# Normalize feature to prevent overflow
x_mean = x.mean()
x_std = x.std()
x_norm = (x - x_mean) / x_std

x = x_norm  # use normalized values for training

# Initialize weights
m = 0 # slope
b = 0 # y-intercept

# Hyperparameters
learning_rate = 0.01 # How big each step in gradient descent is
epochs = 1000 # Number of times we loop through the data
n = len(y) # Number of data points

def predict(m, x, b):
    return m * x + b

# Mean Squared Error Loss
def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

# Vector pointing in the direction of steepest ascent
def compute_gradients(x, y, y_hat):
    dm = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)
    return dm, db

loss_history = []

for epoch in range(epochs):
    y_hat = predict(m, x, b) # Predict values
    loss = mse(y, y_hat) # Calculate loss
    loss_history.append(loss)
    
    dm, db = compute_gradients(x, y, y_hat) # Compute gradients
    
    # Update weights
    m -= learning_rate * dm
    b -= learning_rate * db
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, m = {m:.4f}, b = {b:.4f}")

# Convert slope/intercept back to original scale
m_orig = m / x_std
b_orig = b - m * x_mean / x_std

# Plot regression line
plt.scatter(df['session_seconds'].values, y, label="Actual Data")
plt.plot(df['session_seconds'].values, m_orig * df['session_seconds'].values + b_orig, color="red", label="Fitted Line")
plt.xlabel("Session Seconds")
plt.ylabel("Session Losses")
plt.title("Session Losses vs Session Length")
plt.legend()
plt.show()

print(f"Slope (m): {m_orig:.4f} → extra losses per second")
print(f"Intercept (b): {b_orig:.4f} → predicted losses at 0 seconds")

# Plot loss over epochs
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Time")
plt.show()
