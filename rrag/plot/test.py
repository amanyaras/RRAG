import matplotlib.pyplot as plt
import numpy as np

# Data for Qwen1.5-14B
categories = ['Expected Opinion:\nSupport', 'Expected Opinion:\nOppose']
before_support_qwen = [0.2, 0.9]
before_neutral_qwen = [0.6, 0.1]
before_oppose_qwen = [0.2, 0.0]

after_support_qwen = [0.3, 0.7]
after_neutral_qwen = [0.5, 0.1]
after_oppose_qwen = [0.2, 0.2]

# Data for LLAMA3-8B
before_support_llama = [0.2, 0.8]
before_neutral_llama = [0.6, 0.1]
before_oppose_llama = [0.2, 0.1]

after_support_llama = [0.4, 0.6]
after_neutral_llama = [0.4, 0.2]
after_oppose_llama = [0.2, 0.2]

barWidth = 0.25  # Make the bars thinner
yGap = 0.05  # Gap between bars

# Set position of bar on Y axis
r1 = np.arange(len(categories)) * (barWidth + yGap)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)  # Adjust figsize to maintain the same aspect ratio

# Plot Qwen1.5-14B
axes[0, 0].barh(r1, before_oppose_qwen, color='#ADD8E6', edgecolor='grey', height=barWidth, label='Oppose')
axes[0, 0].barh(r1, before_neutral_qwen, left=before_oppose_qwen, color='#D3D3D3', edgecolor='grey', height=barWidth, label='Neutral')
axes[0, 0].barh(r1, before_support_qwen, left=[i+j for i,j in zip(before_oppose_qwen, before_neutral_qwen)], color='#FF6961', edgecolor='grey', height=barWidth, label='Support')

axes[0, 1].barh(r1, after_oppose_qwen, color='#ADD8E6', edgecolor='grey', height=barWidth)
axes[0, 1].barh(r1, after_neutral_qwen, left=after_oppose_qwen, color='#D3D3D3', edgecolor='grey', height=barWidth)
axes[0, 1].barh(r1, after_support_qwen, left=[i+j for i,j in zip(after_oppose_qwen, after_neutral_qwen)], color='#FF6961', edgecolor='grey', height=barWidth)

# Plot LLAMA3-8B
axes[1, 0].barh(r1, before_oppose_llama, color='#ADD8E6', edgecolor='grey', height=barWidth)
axes[1, 0].barh(r1, before_neutral_llama, left=before_oppose_llama, color='#D3D3D3', edgecolor='grey', height=barWidth)
axes[1, 0].barh(r1, before_support_llama, left=[i+j for i,j in zip(before_oppose_llama, before_neutral_llama)], color='#FF6961', edgecolor='grey', height=barWidth)

axes[1, 1].barh(r1, after_oppose_llama, color='#ADD8E6', edgecolor='grey', height=barWidth)
axes[1, 1].barh(r1, after_neutral_llama, left=after_oppose_llama, color='#D3D3D3', edgecolor='grey', height=barWidth)
axes[1, 1].barh(r1, after_support_llama, left=[i+j for i,j in zip(after_oppose_llama, after_neutral_llama)], color='#FF6961', edgecolor='grey', height=barWidth)

# Add vertical dashed lines at x=0.5
for a in axes.flat:
    a.axvline(x=0.5, color='grey', linestyle='--')

# Titles and Labels
axes[0, 0].set_yticks(r1)
axes[0, 0].set_yticklabels(categories, fontsize=14)
axes[0, 0].set_xlabel('Before manipulation', fontsize=14)
axes[0, 1].set_xlabel('After manipulation', fontsize=14)
axes[1, 0].set_yticks(r1)
axes[1, 0].set_yticklabels(categories, fontsize=14)
axes[1, 0].set_xlabel('Before manipulation', fontsize=14)
axes[1, 1].set_xlabel('After manipulation', fontsize=14)

# Set x and y limits
for a in axes.flat:
    a.set_xlim(0, 1)
    a.set_ylim(-barWidth, r1[-1] + barWidth + yGap)

fig.suptitle('Distribution Variation of RAG Response Opinions', fontsize=16)

# Subplot titles
axes[0, 0].set_title('Based on Qwen1.5-14B', fontsize=14)
axes[0, 1].set_title('Based on Qwen1.5-14B', fontsize=14)
axes[1, 0].set_title('Based on LLAMA3-8B', fontsize=14)
axes[1, 1].set_title('Based on LLAMA3-8B', fontsize=14)

# Adjust space between plots
plt.subplots_adjust(wspace=0.3, hspace=0.4)

# Add legend
axes[0, 0].legend(loc='upper right', fontsize=12)

# Save plot
plt.savefig('/root/workspace/RRAG/rrag/plot/test.png')

# Show plot
plt.show()
