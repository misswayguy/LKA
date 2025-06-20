import matplotlib.pyplot as plt

# Data for plotting
methods = [
    'Normtuning', 'Adapter', 'ST-Adapter', 'Convpass', 'CIAT', 'AIM', 
    'LOSSLESS ADAPTATION', 'RepAdapter', 'Adapterformer', 'VPT', 'LoRA', 
    'Bitfit', 'LKA (Ours)'
]
params = [
    0.003, 0.153, 0.167, 0.155, 0.230, 0.234, 0.153, 0.221, 0.153, 0.532, 
    0.291, 0.007, 0.163
]
accuracy = [
    0.887, 0.903, 0.790, 0.935, 0.919, 0.871, 0.903, 0.919, 0.905, 0.855, 
    0.903, 0.903, 0.952
]

# Corresponding colors and markers
colors = [
    'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'blue', 
    'yellow', 'lime', 'magenta', 'red'
]
markers = [
    '^', '>', '<', 'v', '8', 's', 'P', 'X', 'd', 'p', '*', 'H', '*'
]

# Plotting
plt.figure(figsize=(12, 6))

for i, method in enumerate(methods):
    if method == 'LKA (Ours)':
        plt.scatter(params[i], accuracy[i], color='red', label=method, s=350, marker='*')
        plt.text(params[i] + 0.01, accuracy[i] + 0.002, 'LKA (Ours)', fontsize=12, ha='right', color='red')
    else:
        plt.scatter(params[i], accuracy[i], color=colors[i], label=method, s=300, marker=markers[i])

# Adding the horizontal line for Full Fine-tuning
plt.axhline(y=0.935, color='blue', linestyle='--')
plt.text(0.15, 0.936, 'Full Fine-tuning', color='blue', ha='left', va='bottom', fontsize=10)

# Adding labels and title
plt.xlabel('Trainable Parameters (Millions)')
plt.ylabel('Accuracy')

# Adding the legend
plt.legend(loc='lower right', fontsize='medium', ncol=2)

# Displaying the plot
plt.grid(True)
plt.show()

# Saving the plot
plt.savefig("/home/lusiyuan/ZZQ/prompt/sw/v3_COMPARSION.png", bbox_inches='tight')
