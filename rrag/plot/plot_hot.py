import matplotlib.pyplot as plt
import numpy as np

# Set up data for contour plots
x = np.linspace(-3.0, 3.0, 100)
y = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(x, y)
Z1 = np.sin(X**2 + Y**2)
Z2 = np.cos(X**2 + Y**2)

# Random positions for markers
np.random.seed(0)
init_positions = np.random.randn(2, 2)
clean_positions = np.random.randn(2, 2)
backdoor_positions = np.random.randn(2, 2)

# Set up data for line plot
ratios = np.linspace(0, 1, 10)
acc = np.random.rand(10) * 0.4 + 0.6
asr = np.random.rand(10) * 0.2 + 0.8

# Create figure and axes
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Contour plot 1
contour1 = ax1.contourf(X, Y, Z1, cmap='RdYlBu')
ax1.contour(X, Y, Z1, colors='black')
ax1.plot(init_positions[0, 0], init_positions[0, 1], 'b*', markersize=10, label='Init')
ax1.plot(clean_positions[0, 0], clean_positions[0, 1], 'bo', markersize=10, label='Clean')
ax1.plot(backdoor_positions[0, 0], backdoor_positions[0, 1], 'bs', markersize=10, label='Backdoor')
ax1.set_title('(a) Trigger Word (SST-2).')
ax1.legend()

# Contour plot 2
contour2 = ax2.contourf(X, Y, Z2, cmap='RdYlBu')
ax2.contour(X, Y, Z2, colors='black')
ax2.plot(init_positions[1, 0], init_positions[1, 1], 'b*', markersize=10, label='Init')
ax2.plot(clean_positions[1, 0], clean_positions[1, 1], 'bo', markersize=10, label='Clean')
ax2.plot(backdoor_positions[1, 0], backdoor_positions[1, 1], 'bs', markersize=10, label='Backdoor')
ax2.set_title('(b) Sentence (Scratch, QNLI, size=64).')
ax2.legend()

# Line plot
ax3.plot(ratios, acc, 'b-o', label='ACC')
ax3.plot(ratios, asr, 'b--o', label='ASR')
ax3.set_xlabel('ratio')
ax3.set_ylabel('ACC/ASR')
ax3.set_title('(c) Sentence (Scratch, QNLI, size=64).')
ax3.legend()

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('/root/workspace/RRAG/rrag/plot/Combined_Plot_Random.png')

# Show plot
plt.show()
