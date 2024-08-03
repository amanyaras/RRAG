import matplotlib.pyplot as plt

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

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Create horizontal bars with a consistent color scheme
bars = ax.barh(models, runtimes, color='skyblue', edgecolor='black', alpha=0.7)

# Add text annotations for the bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, f'{width}', ha='left', va='center', fontsize=12, color='black')

# Set labels and title with appropriate font size
ax.set_xlabel('Runtime (s)', fontsize=14, labelpad=10)
ax.set_title('Model Runtimes Comparison', fontsize=16, pad=20)

# Invert y-axis to have the first bar at the top
ax.invert_yaxis()

# Remove frame and add grid for better readability
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.set_tick_params(width=0)
ax.yaxis.set_tick_params(width=0)
ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

# Adjust layout to make room for labels
plt.tight_layout()

# Save plot
plt.savefig('/root/workspace/RRAG/rrag/plot/Model_Runtimes.png')

# Show plot
plt.show()
