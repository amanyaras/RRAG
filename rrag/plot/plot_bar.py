import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Expected Opinion:\nSupport', 'Expected Opinion:\nOppose']
before_support = [2, 9]
before_neutral = [6, 1]
before_oppose = [2, 1]

after_support = [3, 7]
after_neutral = [5, 1]
after_oppose = [2, 2]

barWidth = 1.16  # Make the bars thinner
yGap = 0.33  # Gap between bars

# Set position of bar on X axis
# r1 = np.arange(len(categories)) * (barWidth + yGap)
# print(r1)
r1 = [0.74, 2.19] # 2个柱在y轴的位置
# Plotting
fig, ax = plt.subplots(1, 2, figsize=(29.8, 6.64), sharey=True)  # Make the plot wider

# Plot before manipulation
ax[0].barh(r1, before_oppose, color='#ADD8E6', edgecolor='grey', height=barWidth, label='Oppose')
ax[0].barh(r1, before_neutral, left=before_oppose, color='#D3D3D3', edgecolor='grey', height=barWidth, label='Neutral')
ax[0].barh(r1, before_support, left=[i+j for i,j in zip(before_oppose, before_neutral)], color='#FF6961', edgecolor='grey', height=barWidth, label='Support')

# Plot after manipulation
ax[1].barh(r1, after_oppose, color='#ADD8E6', edgecolor='grey', height=barWidth)
ax[1].barh(r1, after_neutral, left=after_oppose, color='#D3D3D3', edgecolor='grey', height=barWidth)
ax[1].barh(r1, after_support, left=[i+j for i,j in zip(after_oppose, after_neutral)], color='#FF6961', edgecolor='grey', height=barWidth)

# Add vertical dashed lines at x=0.5
for a in ax:
    a.axvline(x=5, color='grey', linestyle='--')


# Titles and Labels
ax[0].set_yticks(r1)
ax[0].set_yticklabels(categories, fontsize=25)  # Increase font size for y-tick labels
ax[0].set_xlabel('Before manipulation', fontsize=25)
ax[1].set_xlabel('After manipulation', fontsize=25)


ax[0].set_xlim(0, 10)
ax[1].set_xlim(0, 10)
ax[0].set_ylim(0, 2.84)
ax[1].set_ylim(0, 2.84)

fig.suptitle('Distribution Variation of RAG Response Opinions Based on Qwen1.5-14B')

# Adjust space between plots
# plt.subplots_adjust(wspace=0.3)

# Add legend
ax[0].legend(loc='upper right')

# Save plot
plt.savefig('/root/workspace/RRAG/rrag/plot/Distribution_Variation_RAG_Response_Opinions.png')

# Show plot
plt.show()
