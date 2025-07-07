import matplotlib.pyplot as plt
import numpy as np

# Generate input values
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))

# Plot the sigmoid function
plt.figure(figsize=(8, 5))
plt.plot(z, sigmoid, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$', color='blue', linewidth=2)
plt.title("Sigmoid Function: Output in (0, 1) and Convexity of the Loss")
plt.xlabel("Logit (z)")
plt.ylabel("Probability")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axhline(1, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
