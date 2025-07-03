import matplotlib.pyplot as plt
import numpy as np

# Create a toy example of a 1D function: f(x) = sin(x)
np.random.seed(0)
x = np.linspace(0, 10, 100)
y_true = np.sin(x)

# Simulate boosting as a sequence of weak learners improving residuals
# Each "stage" will be plotted
stage1 = 0.5 * np.ones_like(x)                     # weak learner: predicts mean
residual1 = y_true - stage1

stage2 = stage1 + 0.3 * np.cos(x)                  # adds more structure
residual2 = y_true - stage2

stage3 = stage2 + 0.2 * np.sin(x) * (x < 5)        # further refines left half
final_prediction = stage3

stage4 = final_prediction + 0.1 * np.sin(2 * x) * (x > 5)
residual4 = y_true - stage4

stage5 = stage4 + 0.05 * np.cos(3 * x)
final_boosted = stage5

# Plotting updated with all five stages
plt.figure(figsize=(12, 6))
plt.plot(x, y_true, label='True Function (sin(x))', linewidth=2, color='black')
plt.plot(x, stage1, label='Stage 1: Constant Prediction', linestyle='--')
plt.plot(x, stage2, label='Stage 2: Adds Cosine Correction', linestyle='--')
plt.plot(x, stage3, label='Stage 3: Focus on Left Residuals', linestyle='--')
plt.plot(x, stage4, label='Stage 4: Refine Right Side', linestyle='--')
plt.plot(x, final_boosted, label='Stage 5: Fine Tuning with Cosine', linestyle='--')

plt.title('Boosting: Sequential Improvement via Residual Learning')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()