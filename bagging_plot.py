import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic "true" function and training data
np.random.seed(1)
x = np.linspace(0, 10, 100)
y_true = np.sin(x)

# Simulate 5 overfitting base learners (e.g., decision trees on bootstrapped data)
# Each with random noise to simulate overfitting to a subset
base_learners = [y_true + np.random.normal(0, 0.3, size=len(x)) for _ in range(5)]

# Averaged prediction (bagging result)
bagged_prediction = np.mean(base_learners, axis=0)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x, y_true, label='True Function (sin(x))', linewidth=2, color='black')

# Plot base learners
for i, learner in enumerate(base_learners):
    plt.plot(x, learner, linestyle='--', linewidth=1, label=f'Base Learner {i+1}', alpha=0.7)

# Plot bagged model
plt.plot(x, bagged_prediction, label='Bagged Prediction (Averaged)', color='red', linewidth=2)

plt.title('Bagging: Averaging Overfitted Learners to Improve Generalization')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
