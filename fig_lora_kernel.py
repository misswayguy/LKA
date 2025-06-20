import matplotlib.pyplot as plt

# Data for plotting
methods = [
    'LoRA r=1', 'LoRA r=2', 'LoRA r=4', 'LoRA r=8', 'LoRA r=16' , 'LoRA r=64'
]
params = [
    0.073, 0.145, 0.291, 0.582, 1.164, 4.672
]
accuracy = [
    0.846, 0.856, 0.866, 0.869, 0.875, 0.880
]

# Data for kernel size
kernel_sizes = [
    'None', '3×3', '5×5', '7×7'
]
kernel_params = [
    0.153, 0.155, 0.158, 0.163
]
kernel_accuracy = [
    0.851, 0.875, 0.881, 0.885
]

# Plotting
plt.figure(figsize=(12, 6))

# Plotting LoRA data
plt.plot(params, accuracy, 'bo-', label='LoRA')
for i, method in enumerate(methods):
    plt.text(params[i], accuracy[i], method, fontsize=10, ha='right')

# Plotting Kernel Size data
plt.plot(kernel_params, kernel_accuracy, 'r^--', label='Kernel Size')
for i, ks in enumerate(kernel_sizes):
    plt.text(kernel_params[i], kernel_accuracy[i], ks, fontsize=10, ha='left')

# Adding labels and title
plt.xlabel('Trainable Parameters (M)')
plt.ylabel('Accuracy')
plt.title('Cell')

# Adding the legend
plt.legend(loc='lower right', fontsize='medium', ncol=1)

# Displaying the plot
plt.grid(True)
plt.show()

# Saving the plot
plt.savefig("/home/lusiyuan/ZZQ/prompt/sw/v2_cell.png", bbox_inches='tight')
