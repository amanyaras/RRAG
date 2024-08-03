import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Generate random data for the boxplots
np.random.seed(0)
data = {
    'Method': ['MPAT'] * 10 + ['BPAT'] * 10 + ['SAFER'] * 10,
    'ASR': np.concatenate([np.random.rand(10) * 0.1 + 0.15,
                           np.random.rand(10) * 0.05 + 0.25,
                           np.random.rand(10) * 0.05 + 0.3]),
    'Dataset': ['IMDB'] * 30
}

data2 = {
    'Method': ['MPAT'] * 10 + ['BPAT'] * 10 + ['SAFER'] * 10,
    'ASR': np.concatenate([np.random.rand(10) * 0.1 + 0.1,
                           np.random.rand(10) * 0.05 + 0.2,
                           np.random.rand(10) * 0.05 + 0.25]),
    'Dataset': ['AGNEWS'] * 30
}

data3 = {
    'Method': ['MPAT'] * 10 + ['BPAT'] * 10 + ['SAFER'] * 10,
    'ASR': np.concatenate([np.random.rand(10) * 0.1 + 0.1,
                           np.random.rand(10) * 0.05 + 0.2,
                           np.random.rand(10) * 0.05 + 0.25]),
    'Dataset': ['SNLI'] * 30
}

df1 = pd.DataFrame(data)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)

# Create the figure and axes
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot the boxplots
sns.boxplot(ax=axes[0], x='Method', y='ASR', data=df1, palette=['blue', 'green', 'orange'])
sns.stripplot(ax=axes[0], x='Method', y='ASR', data=df1, color='black', size=8, jitter=True, dodge=True)
axes[0].set_title('IMDB')
axes[0].set_xlabel('Three different defense methods')
axes[0].set_ylabel('Attack Success Rate (ASR)')

sns.boxplot(ax=axes[1], x='Method', y='ASR', data=df2, palette=['blue', 'green', 'orange'])
sns.stripplot(ax=axes[1], x='Method', y='ASR', data=df2, color='black', size=8, jitter=True, dodge=True)
axes[1].set_title('AGNEWS')
axes[1].set_xlabel('Three different defense methods')
axes[1].set_ylabel('')

sns.boxplot(ax=axes[2], x='Method', y='ASR', data=df3, palette=['blue', 'green', 'orange'])
sns.stripplot(ax=axes[2], x='Method', y='ASR', data=df3, color='black', size=8, jitter=True, dodge=True)
axes[2].set_title('SNLI')
axes[2].set_xlabel('Three different defense methods')
axes[2].set_ylabel('')

# Add subplot labels
axes[0].text(-0.3, 0.35, '(a)', fontsize=16, ha='center', va='center', transform=axes[0].transAxes)
axes[1].text(-0.3, 0.35, '(b)', fontsize=16, ha='center', va='center', transform=axes[1].transAxes)
axes[2].text(-0.3, 0.35, '(c)', fontsize=16, ha='center', va='center', transform=axes[2].transAxes)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('/root/workspace/RRAG/rrag/plot/Boxplot_Comparison.png')

# Show plot
plt.show()
