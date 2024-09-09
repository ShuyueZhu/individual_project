import matplotlib.pyplot as plt
import np
import numpy as np

# # 模型类型和对应的 FID 分数
# # models = ['CycleGAN', 'Latent Diffusion Model']
# # fid_disease = [432.6246462099012, 228.9369933199235]
# # fid_healthy = [470.34986748883045, 186.23208386523143]
#
# models=['Real data', 'Synthetic data from 3D modeling with real data']
# Accuracy=[0.7608695652173914, 0.782608695652174]
# F1=[0.676328502415459, 0.687168610816543]
# # 横坐标位置
# x = np.arange(len(models))*0.5
#
# # 柱子的宽度
# width = 0.15
#
# # 创建图形和子图
# fig, ax = plt.subplots(figsize=(8, 6))  # 设置图形大小
#
# # 绘制柱状图
# # bars1 = ax.bar(x - width/2, fid_disease, width, label='FID for Disease Leaves', color='#5491E2', edgecolor=None)
# # bars2 = ax.bar(x + width/2, fid_healthy, width, label='FID for Healthy Leaves', color='#D63F1C', edgecolor=None)
# bars1 = ax.bar(x - width / 2, Accuracy, width, label='Accuracy', color='#5491E2', edgecolor=None)
# bars2 = ax.bar(x + width / 2, F1, width, label='F1 Score', color='#D63F1C', edgecolor=None)
#
# # # 添加标签和标题
# # ax.set_xlabel('Model Type', fontsize=14, fontweight='bold')
# # ax.set_ylabel('FID Score', fontsize=14, fontweight='bold')
# ax.set_xlabel('Data Type', fontsize=14, fontweight='bold')
# ax.set_ylabel('Score', fontsize=14, fontweight='bold')
# # ax.set_title('FID Scores for Different Models', fontsize=16, fontweight='bold')
# ax.set_xticks(x)
# ax.set_xticklabels(models, fontsize=12)  # 横坐标标签平行于坐标轴
# ax.legend(loc='center left', bbox_to_anchor=(0, 1.05))
#
# # 添加数据标签
# def add_labels(bars):
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height:.2f}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=10, color='black')
#
# add_labels(bars1)
# add_labels(bars2)
#
# # 去掉背景虚线框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# # 设置网格线
# ax.yaxis.grid(True, linestyle='--', alpha=0.7)
#
# # 调整布局，防止标签被截断
# plt.tight_layout()
#
# # 显示图形
# plt.show()
#
# # 保存图形
# # plt.savefig('fid_scores.png', dpi=300, bbox_inches='tight')


# 合成数据的比例
models = [
    'Only synthetic\n data',
    'Real data',
    '0.2 times\nsynthetic data',
    '0.5 times\nsynthetic data',
    '1.0 times\nsynthetic data',
    '1.83 times\nsynthetic data',
    '3 times\n synthetic\nhealthy leaves\n+ all synthetic\ndisease leaves',
    '4 times\n synthetic\nhealthy leaves\n+ all synthetic\ndisease leaves',
    '5 times\n synthetic\nhealthy leaves\n+ all synthetic\ndisease leaves',
    '6 times\n synthetic\nhealthy leaves\n+ all synthetic\ndisease leaves',
    '7 times\n synthetic\nhealthy leaves\n+ all synthetic\ndisease leaves'
    # 'All synthetic\ndata'
]

Accuracy = [
    0.4782608695652174, 0.7608695652173914, 0.7608695652173914, 0.7608695652173914, 0.8043478260869565, 0.782608695652174,
    0.8260869565217391,
    0.8260869565217391, 0.8478260869565217, 0.8478260869565217, 0.8260869565217391
]

F1 = [
    0.4901185770750988, 0.676328502415459, 0.676328502415459, 0.676328502415459, 0.7780914737436476, 0.7210144927536232, 0.8187752970361666,
    0.8187752970361666, 0.8374424552429668, 0.8374424552429668, 0.8187752970361666
]


# 横坐标位置
x = np.arange(len(models)) * 1.5

# 创建图形和子图
fig, ax = plt.subplots(figsize=(17, 7))  # 设置图形大小

# 绘制折线图
line1, = ax.plot(x, Accuracy, marker='o', label='Accuracy', color='#5491E2', linewidth=2, markersize=8)
line2, = ax.plot(x, F1, marker='s', label='F1 Score', color='#D63F1C', linewidth=2, markersize=8)

# 添加标签和标题
ax.set_xlabel('Ratio of Synthetic Data', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)  # 横坐标标签平行于坐标轴
ax.legend(loc='center left', bbox_to_anchor=(0, 1))  # 图例放在右侧

# 添加数据标签
def add_labels(x, y, ax):
    for i in range(len(x)):
        ax.text(x[i], y[i] + 0.01, f'{y[i]:.2f}', ha='center', va='bottom', fontsize=10, color='black')

add_labels(x, Accuracy, ax)
add_labels(x, F1, ax)

# 去掉背景虚线框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# 调整布局，防止标签被截断
plt.tight_layout()

# 显示图形
plt.show()

