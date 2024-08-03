import matplotlib.pyplot as plt

# Data for the plot
steps = [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
bert_accuracy = [80.5, 81.0, 82.0, 82.5, 83.0, 83.5, 83.7, 84.0, 84.2, 84.5]
alum_accuracy = [80.0, 81.5, 82.5, 83.0, 83.5, 84.0, 84.5, 85.0, 85.5, 86.0]

# Create the plot with improved aesthetics
plt.figure(figsize=(10, 5))
plt.plot(steps, bert_accuracy, marker='o', linestyle='-', color='#1f77b4', label='BERT', linewidth=2, markersize=8)
plt.plot(steps, alum_accuracy, marker='s', linestyle='-', color='#ff7f0e', label='ALUM', linewidth=2, markersize=8)

# Adding labels and title
plt.xlabel('Number of pre-training steps', fontsize=14)
plt.ylabel('Accuracy (MNLI)', fontsize=14)
plt.title('Accuracy vs Number of pre-training steps', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot to a specified path
save_path = '/root/workspace/RRAG/rrag/plot/accuracy_plot.png'
plt.savefig(save_path, bbox_inches='tight')

# Show the plot
plt.show()
