import matplotlib.pyplot as plt

# Data
methods = ["Full", "Normtuning", "Prompt", "LoRA", "Bitfit", "Adapter", "Adapterformer", "PUMA"]
trainable_params = [28.288, 0.004, 0.813, 2.270, 0.162, 3.055, 1.334, 3.818]  # in millions
accuracy = [
    [0.924, 0.851, 0.900, 0.984, 0.935],
    [0.836, 0.841, 0.767, 0.914, 0.887],
    [0.837, 0.838, 0.896, 0.969, 0.855],
    [0.866, 0.854, 0.900, 0.988, 0.903],
    [0.855, 0.851, 0.867, 0.981, 0.903],
    [0.851, 0.784, 0.900, 0.978, 0.925],
    [0.869, 0.863, 0.900, 0.988, 0.919],
    [0.883, 0.870, 0.933, 0.989, 0.952]
]

# Data labels
datasets = ["Blood cell", "Breast Ultrasound", "Brain tumor", "Tuberculosis", "COVID-19"]
colors = ['red', 'green', 'blue', 'orange', 'purple']
shapes = ['o', 's', 'D', '^', 'v', '<', '>', '*']  # PUMA is represented by '*'

# Plot
plt.figure(figsize=(12, 8))

# Plot each method's accuracy for each dataset
for i, method in enumerate(methods):
    for j, acc in enumerate(accuracy[i]):
        plt.scatter(trainable_params[i], acc, color=colors[j], marker=shapes[i], s=100, label=f'{method} - {datasets[j]}' if j == 0 else "")

# Plot horizontal lines for Full method results
for j, dataset in enumerate(datasets):
    plt.axhline(y=accuracy[0][j], color=colors[j], linestyle='--', linewidth=1)

# Legend for the shapes in the plot
shape_legend = [plt.Line2D([0], [0], marker=shapes[i], color='w', markerfacecolor='gray', markersize=10, label=methods[i]) for i in range(len(methods))]
color_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[j], markersize=10, label=datasets[j]) for j in range(len(datasets))]

# Combine legends
plt.legend(handles=shape_legend + color_legend, loc='lower right', bbox_to_anchor=(1, 0))

plt.xlabel('Number of Trainable Parameters (M)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title(' Performance of PUMA compared to other methods', fontsize=14)
plt.grid(True)

# Set x-axis starting from 0
plt.xlim(left=0)

# Save the plot
plt.savefig("v7_accuracy_vs_trainable_params_adjusted.png", bbox_inches='tight')

# Display the plot
plt.show()
