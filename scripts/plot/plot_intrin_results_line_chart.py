import numpy as np
import matplotlib.pyplot as plt

# ticks = [5,10,15,20,25,30]
# data_lists = [
#     [98.99,99.29,99.30,99.25,99.23,99.21],
#     [97.38,96.61,96.05,95.63,95.44,95.08],
#     [97.69,97.51,97.29,97.07,97.00,96.84]
# ]
# label_lists = [
#     "Control",
#     "Data",
#     "Overall"
# ]
# color_lists = ['red', 'royalblue', 'yellowgreen']
# marker_lists = ['s','s','s','s']
# path = "F:/坚果云文件/我的坚果云/研三上/论文材料/pdbert_intrin_partial_results.jpg"

ticks = ['(0,10]','(10,20]','(20,30]','(30,+∞)']
data_lists = [
    [99.91,99.82,99.44,99.23],
    [97.59,96.35,95.28,94.31],
    [98.07,97.56,96.93,96.46]
]
label_lists = [
    "Control",
    "Data",
    "Overall"
]
color_lists = ['red', 'royalblue', 'yellowgreen']
marker_lists = ['s','s','s','s']
path = "F:/坚果云文件/我的坚果云/研三上/论文材料/pdbert_intrin_full_results.jpg"

marker_size = 8
title = ''
x_title = 'Partial Code LOC'
y_title = 'F1(%)'

fig_size = (6,4)
dpi = 300

plt.figure(dpi=dpi, figsize=fig_size)
plt.xticks(np.arange(len(ticks)), ticks)
# plt.xticks(ticks)
plt.title(title)
plt.xlabel(x_title)
plt.ylabel(y_title)
plt.grid(True,axis='y')

for data,label,color,marker in zip(data_lists,label_lists,color_lists,marker_lists):
    plt.plot(ticks, data, color=color, marker=marker, label=label, markersize=marker_size)

plt.legend()
# plt.show()
plt.savefig(fname=path, dpi=300)

