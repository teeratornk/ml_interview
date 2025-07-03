import matplotlib.pyplot as plt
import numpy as np

# Simulate model complexity
complexity = np.linspace(0, 10, 100)

# Define arbitrary functions for bias^2, variance, irreducible error
bias_squared = np.exp(-0.5 * complexity) * 2
variance = np.exp(0.5 * (complexity - 5)) * 0.1
irreducible_error = np.ones_like(complexity) * 0.5

# Total error is sum of all three
total_error = bias_squared + variance + irreducible_error

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(complexity, bias_squared, label='Bias²', linestyle='--')
plt.plot(complexity, variance, label='Variance', linestyle='--')
plt.plot(complexity, irreducible_error, label='Irreducible Error', linestyle='--')
plt.plot(complexity, total_error, label='Total Error', linewidth=2)

plt.title('Bias-Variance Trade-off and Model Complexity')
plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
