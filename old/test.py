import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import os
import re
import warnings
from pathlib import Path
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
# 议题名称
themes = [
    "食品安全与营养争议",
    "消费者与企业家的对立立场",
    "产品供应链真实性质疑",
    "社会发展隐喻争议",
    "个体消费抗拒行为",
    "历史创伤与信任危机",
    "语言暴力与身份污名化",
    "国际品牌参照下的预期管理",
    "技术风险感知"
]

# 完整距离矩阵 (9x9，基于 d=(1-r)/2)
dist_mat = np.array([
    [0.00, 0.78, 0.57, 0.62, 0.45, 0.52, 0.62, 0.75, 0.57],
    [0.78, 0.00, 0.45, 0.55, 0.65, 0.51, 0.45, 0.40, 0.61],
    [0.57, 0.45, 0.00, 0.53, 0.59, 0.52, 0.26, 0.49, 0.49],
    [0.62, 0.55, 0.53, 0.00, 0.76, 0.52, 0.56, 0.37, 0.47],
    [0.45, 0.65, 0.59, 0.76, 0.00, 0.47, 0.52, 0.70, 0.66],
    [0.52, 0.51, 0.52, 0.52, 0.47, 0.00, 0.50, 0.49, 0.49],
    [0.62, 0.45, 0.26, 0.56, 0.52, 0.50, 0.00, 0.41, 0.49],
    [0.75, 0.40, 0.49, 0.37, 0.70, 0.49, 0.41, 0.00, 0.50],
    [0.57, 0.61, 0.49, 0.47, 0.66, 0.49, 0.49, 0.50, 0.00]
])

# 压缩为距离向量
dist_vec = squareform(dist_mat)

# 层次聚类（平均连接法）
Z = linkage(dist_vec, method='average')
def setup_chinese_font():
    font_path = "/root/pmd/SIMSUN.TTC"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
setup_chinese_font()

plt.figure(figsize=(12, 6))
dendrogram(Z, labels=themes, orientation='top', leaf_rotation=45, leaf_font_size=10)
plt.title('议题层次聚类树状图（基于相关系数距离，平均连接法）', fontsize=14)
plt.xlabel('议题', fontsize=12)
plt.ylabel('距离 d = (1 - r)/2', fontsize=12)
plt.tight_layout()

# 保存图片（可选）
plt.savefig('dendrogram.png', dpi=300)
plt.show()