import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Generate random data for the plots
np.random.seed(0)
x = np.random.rand(30) * 2
y1 = -0.2 * x + 0.1 * np.random.randn(30)
y2 = 0.2 * x - 0.1 * np.random.randn(30)
y3 = np.random.rand(30) * 0.3

# Create a DataFrame
data = pd.DataFrame({
    'avg_learnability': x,
    'robustness': y1,
    'post_aug_delta': y2,
    'third_metric': y3
})

# Set up the figure and axes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: avg_learnability vs robustness
sns.regplot(ax=axes[0], x='avg_learnability', y='robustness', data=data, scatter_kws={'s': 50})
axes[0].set_title(r'$\rho = -0.821^*$', fontsize=12)
axes[0].set_xlabel('avg learnability')
axes[0].set_ylabel('robustness')

# Plot 2: avg_learnability vs post_aug_delta
sns.regplot(ax=axes[1], x='avg_learnability', y='post_aug_delta', data=data, scatter_kws={'s': 50})
axes[1].set_title(r'$\rho = 0.846^*$', fontsize=12)
axes[1].set_xlabel('avg learnability')
axes[1].set_ylabel('post aug $\Delta$')

# Plot 3: robustness vs post_aug_delta with color by avg_learnability
scatter = axes[2].scatter(data['robustness'], data['post_aug_delta'], c=data['avg_learnability'], cmap='viridis', s=50)
axes[2].set_xlabel('robustness')
axes[2].set_ylabel('post aug $\Delta$')
cbar = plt.colorbar(scatter, ax=axes[2])
cbar.set_label('avg learnability')

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('/root/workspace/RRAG/rrag/plot/Triple_Plot.png')

# Show plot
plt.show()
