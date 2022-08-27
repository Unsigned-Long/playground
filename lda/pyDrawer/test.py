from array import array
from posixpath import split
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


import csv

plt.rcParams["font.family"] = "Ubuntu Mono"
plt.rcParams["font.size"] = 13

filename = "./data/data.csv"


# 特征向量为4维的
X = []
# 标签值用于将样本显示成不同的颜色
y = []

with open(filename) as file:
    lines = csv.reader(file)
    for line in lines:
        if line[0] == "Q1":
            idx = 0
        elif line[0] == "Q2":
            idx = 1
        elif line[0] == "Q3":
            idx = 2
        elif line[0] == "Q4":
            idx = 3
        ary = []
        for i in line[1:]:
            ary.append(float(i))

        X.append(ary)
        y.append(idx)


lda = LinearDiscriminantAnalysis(n_components=2)

X_r = lda.fit(X, y).transform(X)

print(lda.get_params())


target_names = ['Q1', 'Q2', 'Q3', 'Q4']
colors = ['navy', 'turquoise', 'darkorange', 'red']

plt.figure()
# 显示降维后的样本
for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
    x1_ary = []
    x2_ary = []
    for idx in range(0, len(y)):
        if(y[idx] == i):
            x1_ary.append(X_r[idx][0])
            x2_ary.append(X_r[idx][1])

    plt.scatter(x1_ary, x2_ary,
                alpha=.8, color=color, label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('sfasfa')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(ls='--', alpha=0.5)

for i in range(0, len(y)):
    print(y[i], X_r[i][0], X_r[i][1], sep=',')

plt.savefig("./img/fig.png", dpi=1200)
