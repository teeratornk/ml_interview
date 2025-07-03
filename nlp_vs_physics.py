import pandas as pd
import matplotlib.pyplot as plt

# Define the data
data = {
    "Property": [
        "Primary training data",
        "Typical data point size",
        "Governing constraints/laws",
        "Benchmark availability",
        "Generalization target",
        "Fine-tuning cost for new tasks",
        "Interpretability demand",
        "Formal error guarantees",
        "Data generation cost"
    ],
    "NLP Foundation Models": [
        "Web text, books, code, etc.",
        "Token embedding (~1k floats)",
        "Human grammar & semantics (soft)",
        "High (e.g., WikiText, Pile)",
        "Across linguistic tasks/domains",
        "Low (prompting / few-shot)",
        "Medium (bias, toxicity audits)",
        "Rare; primarily empirical",
        "Low (scraped from web)"
    ],
    "Computational‑Science Foundation Models": [
        "Simulation & experimental fields",
        "Large 3‑D fields (MB–GB each)",
        "Physical laws (conservation, symmetry)",
        "Low (few multi‑physics corpora)",
        "Across physics, geometries, scales",
        "High (re-solve / physics tuning)",
        "High (trust in safety-critical apps)",
        "Expected (stability, error bounds)",
        "High (costly simulations or experiments)"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plot and save table
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

# Create the table
table_data = [
    [row["Property"], row["NLP Foundation Models"], row["Computational‑Science Foundation Models"]]
    for _, row in df.iterrows()
]
column_labels = ["Property", "NLP Foundation Models", "Computational-Science Foundation Models"]

# Add the table to the plot
table = plt.table(
    cellText=table_data,
    colLabels=column_labels,
    loc='center',
    cellLoc='left',
    colWidths=[0.25, 0.35, 0.4]
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Save as PNG
output_path = "foundation_model_comparison.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)

print(f"Saved to {output_path}")
