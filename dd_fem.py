import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Draw local subdomains (e.g., 2x2 elements)
local_positions = [(1, 1), (1, 3), (3, 1), (3, 3)]
for (x, y) in local_positions:
    rect = patches.Rectangle((x, y), 1.5, 1.5, linewidth=1.5, edgecolor='royalblue', facecolor='lightblue')
    ax.add_patch(rect)
    ax.text(x + 0.75, y + 0.75, 'Local\nTraining', ha='center', va='center', fontsize=9)

# Draw arrows to global domain
for (x, y) in local_positions:
    ax.annotate('', xy=(6, 2.5), xytext=(x + 0.75, y + 0.75),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='gray'))

# Draw global domain box
global_box = patches.Rectangle((6.5, 1.5), 2.5, 2.5, linewidth=2, edgecolor='darkgreen', facecolor='palegreen')
ax.add_patch(global_box)
ax.text(7.75, 2.75, 'Global\nAssembly', ha='center', va='center', fontsize=10, weight='bold')

# Add final arrow for PDE solving
ax.annotate('', xy=(9.5, 2.5), xytext=(9, 2.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
ax.text(9.7, 2.5, 'PDE\nSolve', va='center', fontsize=10)

# Save the plot
output_path = "ddfem_local_to_global.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)

print(f"Schematic saved to {output_path}")
