# - Low complexity: underfitting
# - Medium complexity: optimal
# - High complexity: overfitting

complexity = np.linspace(1, 10, 100)
training_error = 1 / (complexity + 0.5)  # Training error decreases with complexity
test_error = training_error + 0.02 * (complexity - 5) ** 2  # Test error follows U-shape

# Plot
plt.figure(figsize=(10, 6))
plt.plot(complexity, training_error, label='Training Error', linewidth=2)
plt.plot(complexity, test_error, label='Test Error', linewidth=2)
plt.axvline(x=5, color='gray', linestyle=':', label='Optimal Complexity')

plt.title('Training vs Test Error and Model Complexity')
plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
