import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
fontsize = 20
linewidth=3
fontsize_x=18
fontsize_tick=18
markersize = '8'
alpha = 0.5
font_label_x={"family":"serif","size":16}
font_label_y={"family":"serif","size":16}
font_tick={"family":"serif","fontsize":10}
fig=figure(num=None, figsize=(6, 3), dpi=120, facecolor='w', edgecolor='k')
#fig=figure(num=None, figsize=(18, 10), dpi=120, facecolor='w', edgecolor='k')  # Adjusted the figsize for more vertical space
plt.rcParams["font.family"] = "serif"
fontsize = 20
markersize = '10'
alpha = 0.6
colors=['#2f4858','#33658a','#86bbd8','#f6ae2d','#f26419']
markers=[':*','-o','--s','-.d','-x']

Label_kernel=["1x1","3X3","5X5","7X7"]
Lable_hiddensize=["8","9","10","11"]

def Set_tickSize(ax,size=14):
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(size)
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
# Data
#kernelsize_V = [0.909*100, 0.919*100, 0.935*100, 0.952*100]
kernelsize_V = [0.826*100, 0.844*100, 0.870*100, 0.876*100]
kernelsize = [0.317*1e6, 0.319*1e6, 0.322*1e6, 0.326*1e6]
# Adapter_size_v = [0.903*100, 0.904*100, 0.905*100, 0.906*100]
Adapter_size_v = [0.784*100, 0.784*100, 0.786*100, 0.790*100]
Adapter_p = [0.317*1e6, 0.355*1e6, 0.390*1e6, 0.427*1e6]

Label_kernel = [r"1$\times$1", r"3$\times$3", r"5$\times$5", r"7$\times$7"]
Lable_hiddensize = ["8", "9", "10", "11"]

# Plot settings
fontsize = 16
linewidth = 3
fontsize_x = 16
fontsize_tick = 12
markersize = '9'
alpha = 0.5
font_label_x = {"family": "serif", "size": 20}
font_label_y = {"family": "serif", "size": 20}
font_tick = {"family": "serif", "fontsize": 12}


fig = figure(num=None, figsize=(6, 3), dpi=120, facecolor='w', edgecolor='k')

colors = ['#2f4858', '#33658a', '#86bbd8', '#f6ae2d', '#f26419']
markers = [':*', '-o', '--s', '-.d', '-x']

# Plotting
vgg_all = fig.add_subplot(1, 1, 1)
vgg_all.plot(kernelsize, kernelsize_V, markers[0], color=colors[4], linewidth=linewidth, markersize=markersize, label=r"LKA")
vgg_all.plot(Adapter_p[:-1], Adapter_size_v[:-1], markers[1], color=colors[1], linewidth=linewidth, markersize=markersize, label=r"Vanilla Adapter")

# Adding annotations for Kernel sizes
for i, label in enumerate(Label_kernel):
    vgg_all.annotate(label, (kernelsize[i], kernelsize_V[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=14)

# Adding annotations for Adapter hidden sizes
for i, label in enumerate(Lable_hiddensize):
    vgg_all.annotate(label, (Adapter_p[i], Adapter_size_v[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=14)

# Customizing the plot
vgg_all.spines['right'].set_visible(False)
vgg_all.spines['top'].set_visible(False)
vgg_all.grid(True, linestyle='-', linewidth=0.5)
# vgg_all.set_xticklabels(["",r"0.32M",r"0.34M",r"0.36M",r"0.38M"])

xticks_positions = [0.32e6, 0.34e6, 0.36e6, 0.38e6]  # 根据你的数据调整刻度值
xtick_labels = [r"0.32M", r"0.34M", r"0.36M", r"0.38M"]

vgg_all.set_title("BUSI Dataset", fontsize=8)


vgg_all.set_xticks(xticks_positions)  # 先设置刻度位置
vgg_all.set_xticklabels(xtick_labels)  # 然后设置刻度标签

vgg_all.set_xlabel('#Trainable Parameters',fontsize=fontsize_x)
vgg_all.set_ylabel(r"ACC(%)($\uparrow$)", fontsize=fontsize)
vgg_all.tick_params(axis='both', which='major', labelsize=fontsize_tick)
# Adding legend
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
fig_legend = fig.legend(loc='lower center', bbox_to_anchor=(0.3, -0.12, 1, 1), fancybox=False, shadow=False, ncol=1, fontsize=15, frameon=False)

# Adjusting the layout
fig.subplots_adjust(left=0.28, bottom=-0.5, right=0.99, top=0.1, wspace=0.2, hspace=0.2)

# Saving the plot
plt.savefig("/home/lusiyuan/ZZQ/prompt/sw/KernelsizeVSParameters_BUSI_2.pdf", bbox_extra_artists=(fig_legend,), bbox_inches='tight')

plt.show()