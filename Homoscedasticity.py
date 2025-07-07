import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(0)
X = np.linspace(0, 10, 100)
y_true = 2 * X + 1

# Create homoscedastic residuals (constant variance)
noise_homo = np.random.normal(0, 1, size=X.shape)
y_homo = y_true + noise_homo

# Create heteroscedastic residuals (increasing variance)
noise_hetero = np.random.normal(0, X * 0.5, size=X.shape)
y_hetero = y_true + noise_hetero

# Fit linear regression models
model_homo = LinearRegression().fit(X.reshape(-1, 1), y_homo)
model_hetero = LinearRegression().fit(X.reshape(-1, 1), y_hetero)

# Get predictions and residuals
pred_homo = model_homo.predict(X.reshape(-1, 1))
residuals_homo = y_homo - pred_homo

pred_hetero = model_hetero.predict(X.reshape(-1, 1))
residuals_hetero = y_hetero - pred_hetero

# Plotting residuals
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axs[0].scatter(pred_homo, residuals_homo, alpha=0.7)
axs[0].axhline(0, color='gray', linestyle='--')
axs[0].set_title("Homoscedastic Residuals")
axs[0].set_xlabel("Predicted Values")
axs[0].set_ylabel("Residuals")

axs[1].scatter(pred_hetero, residuals_hetero, alpha=0.7, color='tomato')
axs[1].axhline(0, color='gray', linestyle='--')
axs[1].set_title("Heteroscedastic Residuals")
axs[1].set_xlabel("Predicted Values")

plt.suptitle("Detecting Homoscedasticity vs Heteroscedasticity via Residual Plots", fontsize=14)
plt.tight_layout()
plt.show()