import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set seed for reproducibility
np.random.seed(42)

# Generate data: true relationship is non-linear
X = np.linspace(0, 10, 100).reshape(-1, 1)
y_true = np.sin(X).ravel()
y_obs = y_true + np.random.normal(0, 0.2, size=y_true.shape)

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y_obs)
y_pred = model.predict(X)

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X.ravel(), y=y_obs, label='Observed Data (with noise)', alpha=0.6)
plt.plot(X, y_true, label='True Relationship (Non-linear)', color='green', linewidth=2)
plt.plot(X, y_pred, label='Linear Regression Fit', color='red', linestyle='--', linewidth=2)

plt.title('Linear Regression vs Non-Linear Ground Truth')
plt.xlabel('Feature X')
plt.ylabel('Target y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()