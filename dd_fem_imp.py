import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Define method labels, positions, and colors
methods = ["Neural Network Basis", "POD Basis", "Symbolic/Sparse Regression Basis"]
positions = [0.5, 4.5, 8.5]
colors = ["skyblue", "lightgray", "lightgreen"]

# Draw rectangles and labels
for label, x, color in zip(methods, positions, colors):
    rect = patches.Rectangle((x, 2), 3, 2.5, linewidth=2, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + 1.5, 4.2, label, ha='center', va='bottom', fontsize=12, weight='bold')

# Add description text below each block
descriptions = [
    "Black-box, flexible\nbut hard to interpret",
    "Compact, data-driven\nbut low fidelity in nonlinear cases",
    "Explicit, interpretable\nand physics-aware"
]
for desc, x in zip(descriptions, positions):
    ax.text(x + 1.5, 1.2, desc, ha='center', fontsize=11)

# Add the main title
ax.text(6, 5, "Comparison of Basis Function Strategies in DD-FEM", ha='center', fontsize=14, weight='bold')

# Save the figure
plt.savefig("ddfem_basis_comparison_improved.png", bbox_inches='tight', dpi=300)
plt.show()