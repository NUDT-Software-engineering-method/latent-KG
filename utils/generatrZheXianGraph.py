# -*- coding: utf-8 -*-
# @Time    : 2022/3/11 上午10:05
# @Author  : WuDiDaBinGe
# @FileName: generatrZheXianGraph.py
# @Software: PyCharm

import matplotlib.pyplot as plt

# 折线图
x = [1, 2, 3, 4, 5, 6, 7]
x_index = [20, 50, 100, 150, 200, 500, 1000]  # 点的横坐标
k1 = [0.49566, 0.49774, 0.50125, 0.48071, 0.491, 0.47856, 0.48183]  # 线1的纵坐标

fig, ax1 = plt.subplots()
_ = plt.xticks(x, x_index)

ax1.plot(x, k1, 's-', color='b', label="TRGKG")

ax1.set_xlabel("Number of Topics", fontsize=15)
ax1.set_ylabel("MAP", fontsize=15)

handles1, labels1 = ax1.get_legend_handles_labels()
plt.legend(handles1, labels1, loc='best', fontsize=15)

plt.savefig("TwitterTopicNumber.pdf", format="pdf")
# 子图2
k2 = [0.41414, 0.40286, 0.40906, 0.41448, 0.4126, 0.39954, 0.3356]  # 线1的纵坐标


fig, ax2 = plt.subplots()
_ = plt.xticks(x, x_index)

ax2.plot(x, k2, 's-', color='r', label="TRGKG")
ax2.set_xlabel("Number of Topics", fontsize=15)
ax2.set_ylabel("MAP", fontsize=15)

handles2, labels2 = ax2.get_legend_handles_labels()
plt.legend(handles2, labels2, loc='best', fontsize=15)
#plt.show()
plt.savefig("StackExchangeTopicNumber.pdf", format="pdf")
