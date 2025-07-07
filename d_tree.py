import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris

# Load a simple dataset
iris = load_iris()
X = iris.data
y = iris.target

# Fit a decision tree classifier (shallow for visual clarity)
clf = DecisionTreeClassifier(max_depth=3, random_state=0)
clf.fit(X, y)

# Plot the tree
plt.figure(figsize=(14, 6))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names,
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree (max_depth=3) Trained on Iris Dataset")
plt.tight_layout()
plt.show()
