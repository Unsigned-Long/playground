import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib

# 载入iris数据集
iris = datasets.load_iris()
# 特征向量为4维的
X = iris.data
# 标签值用于将样本显示成不同的颜色
y = iris.target

target_names = iris.target_names
target_names

print(X)
print(y)

# array(['setosa', 'versicolor', 'virginica'], dtype='<U10')

# 创建LDA降维模型，并计算投影矩阵，对X执行降维操作，得到降维后的结果X_r
lda = LinearDiscriminantAnalysis(n_components=2)
X_r = lda.fit(X, y).transform(X)

print(target_names)

colors = ['navy', 'turquoise', 'darkorange']
plt.figure()
# 显示降维后的样本
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1],
                alpha=.8, color=color, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.show()
