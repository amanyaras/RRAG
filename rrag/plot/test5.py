import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Generate random data for the plots
np.random.seed(0)
data_top_10 = np.random.normal(2, 0.5, 100)
data_bottom_10 = np.random.normal(3, 0.7, 100)
data_pat = np.random.normal(4, 0.6, 100)
data_wo_cons = np.random.normal(6, 0.8, 100)

# Create a DataFrame for Seaborn
data = pd.DataFrame({
    'Top-10': data_top_10,
    'Bottom-10': data_bottom_10,
    'PAT': data_pat,
    'w/o Cons.': data_wo_cons
})

# Create the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Plot for BERT-Large
sns.kdeplot(data['Top-10'], ax=axes[0], label='Top-10', shade=True)
sns.kdeplot(data['Bottom-10'], ax=axes[0], label='Bottom-10', shade=True)
sns.kdeplot(data['PAT'], ax=axes[0], label='PAT', shade=True)
sns.kdeplot(data['w/o Cons.'], ax=axes[0], label='w/o Cons.', shade=True)
axes[0].set_title('BERT-Large')
axes[0].set_xlabel('log PPL')

# Plot for MiniLM-L-12
sns.kdeplot(data['Top-10'], ax=axes[1], label='Top-10', shade=True)
sns.kdeplot(data['Bottom-10'], ax=axes[1], label='Bottom-10', shade=True)
sns.kdeplot(data['PAT'], ax=axes[1], label='PAT', shade=True)
sns.kdeplot(data['w/o Cons.'], ax=axes[1], label='w/o Cons.', shade=True)
axes[1].set_title('MiniLM-L-12')
axes[1].set_xlabel('log PPL')

# Add a legend
axes[0].legend()
axes[1].legend()

# Add a main title
plt.suptitle('Figure 3: Distributions of log perplexity (PPL) calculated by GPT-2 on TREC DL 2019 passages and triggers.', y=-0.05, fontsize=14)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('rrag/plot/Distributions_PPL.png')

# Show plot
plt.show()
