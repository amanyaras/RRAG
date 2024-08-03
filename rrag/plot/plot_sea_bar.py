import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data
models = [
    "RAG-EDA (ours)",
    "RAG-EDA+GPT4",
    "ITER-RETGEN",
    "Llmlingua",
    "HyDE",
    "RAG-fusion",
    "Vanilla-RAG"
]
runtimes = [9.2, 8.26, 14.2, 5.78, 12.61, 8.21, 6.8]

# Set the style
sns.set(style="whitegrid")

# Create a DataFrame
data = pd.DataFrame({
    'Model': models,
    'Runtime (s)': runtimes
})

# Create a horizontal bar plot
plt.figure(figsize=(12, 8))
barplot = sns.barplot(
    x='Runtime (s)', y='Model', data=data, palette='coolwarm', edgecolor='black'
)

# Add text annotations
for index, value in enumerate(runtimes):
    barplot.text(value + 0.3, index, f'{value}', color='black', ha="left", va='center', fontsize=12, weight='bold')

# Set labels and title
plt.xlabel('Runtime (s)', fontsize=14, labelpad=10)
plt.ylabel('')
plt.title('Model Runtimes Comparison', fontsize=18, pad=20, weight='bold')

# Customize ticks and spines
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
barplot.spines['top'].set_visible(False)
barplot.spines['right'].set_visible(False)
barplot.spines['left'].set_visible(False)
barplot.spines['bottom'].set_linewidth(0.5)

# Add a grid for better readability
plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)

# Adjust layout to make room for labels
plt.tight_layout()

# Save plot
plt.savefig('/root/workspace/RRAG/rrag/plot/Model_Runtimes_Seaborn_Styled.png')

# Show plot
plt.show()
