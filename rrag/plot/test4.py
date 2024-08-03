import matplotlib.pyplot as plt
import numpy as np

# Generate some random data for the plots
np.random.seed(0)
original_example = np.random.randn(10, 2)
semantic_similar_example = np.random.randn(10, 2)
adversarial_example = np.random.randn(5, 2)
benign_adversarial_example = np.random.randn(5, 2)
malicious_adversarial_example = np.random.randn(5, 2)
semantic_manifold = np.random.randn(15, 2)

# Create decision boundaries as random curves
t = np.linspace(0, 2 * np.pi, 100)
boundary_original = np.c_[np.cos(t), np.sin(t)]
boundary_rl = np.c_[np.cos(t) + 0.5, np.sin(t) + 0.5]
boundary_at = np.c_[np.cos(t) - 0.5, np.sin(t) - 0.5]
boundary_mpat = np.c_[np.cos(t), np.sin(t) - 0.5]

# Create the figure and axes
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Plot settings
scatter_settings = {'s': 100, 'edgecolor': 'black'}

# Subplot 1
axes[0, 0].scatter(original_example[:, 0], original_example[:, 1], c='green', label='Original example', **scatter_settings)
axes[0, 0].scatter(semantic_similar_example[:, 0], semantic_similar_example[:, 1], c='blue', label='Semantic similar example', **scatter_settings)
axes[0, 0].scatter(adversarial_example[:, 0], adversarial_example[:, 1], c='red', label='Adversarial example', **scatter_settings)
axes[0, 0].scatter(benign_adversarial_example[:, 0], benign_adversarial_example[:, 1], c='orange', label='Benign adversarial example', **scatter_settings)
axes[0, 0].scatter(malicious_adversarial_example[:, 0], malicious_adversarial_example[:, 1], c='brown', label='Malicious adversarial example', **scatter_settings)
axes[0, 0].plot(boundary_original[:, 0], boundary_original[:, 1], c='green')
axes[0, 0].plot(semantic_manifold[:, 0], semantic_manifold[:, 1], 'k--')
axes[0, 0].set_title('(a) Original decision boundary.')

# Subplot 2
axes[0, 1].scatter(original_example[:, 0], original_example[:, 1], c='green', **scatter_settings)
axes[0, 1].scatter(semantic_similar_example[:, 0], semantic_similar_example[:, 1], c='blue', **scatter_settings)
axes[0, 1].scatter(adversarial_example[:, 0], adversarial_example[:, 1], c='red', **scatter_settings)
axes[0, 1].scatter(benign_adversarial_example[:, 0], benign_adversarial_example[:, 1], c='orange', **scatter_settings)
axes[0, 1].scatter(malicious_adversarial_example[:, 0], malicious_adversarial_example[:, 1], c='brown', **scatter_settings)
axes[0, 1].plot(boundary_rl[:, 0], boundary_rl[:, 1], c='blue')
axes[0, 1].plot(semantic_manifold[:, 0], semantic_manifold[:, 1], 'k--')
axes[0, 1].set_title('(b) Decision boundary after RL.')

# Subplot 3
axes[1, 0].scatter(original_example[:, 0], original_example[:, 1], c='green', **scatter_settings)
axes[1, 0].scatter(semantic_similar_example[:, 0], semantic_similar_example[:, 1], c='blue', **scatter_settings)
axes[1, 0].scatter(adversarial_example[:, 0], adversarial_example[:, 1], c='red', **scatter_settings)
axes[1, 0].scatter(benign_adversarial_example[:, 0], benign_adversarial_example[:, 1], c='orange', **scatter_settings)
axes[1, 0].scatter(malicious_adversarial_example[:, 0], malicious_adversarial_example[:, 1], c='brown', **scatter_settings)
axes[1, 0].plot(boundary_at[:, 0], boundary_at[:, 1], c='orange')
axes[1, 0].plot(semantic_manifold[:, 0], semantic_manifold[:, 1], 'k--')
axes[1, 0].set_title('(c) Decision boundary after AT.')

# Subplot 4
axes[1, 1].scatter(original_example[:, 0], original_example[:, 1], c='green', **scatter_settings)
axes[1, 1].scatter(semantic_similar_example[:, 0], semantic_similar_example[:, 1], c='blue', **scatter_settings)
axes[1, 1].scatter(adversarial_example[:, 0], adversarial_example[:, 1], c='red', **scatter_settings)
axes[1, 1].scatter(benign_adversarial_example[:, 0], benign_adversarial_example[:, 1], c='orange', **scatter_settings)
axes[1, 1].scatter(malicious_adversarial_example[:, 0], malicious_adversarial_example[:, 1], c='brown', **scatter_settings)
axes[1, 1].plot(boundary_mpat[:, 0], boundary_mpat[:, 1], c='brown')
axes[1, 1].plot(semantic_manifold[:, 0], semantic_manifold[:, 1], 'k--')
axes[1, 1].set_title('(d) Decision boundary after MPAT.')

# Add a legend
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=1, fontsize=12, frameon=True)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('/root/workspace/RRAG/rrag/plot/Decision_Boundaries.png')

# Show plot
plt.show()
