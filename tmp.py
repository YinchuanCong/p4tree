import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 示例数据: 3个样本, 2个特征
X = np.array([[1, 2],
              [2, 3],
              [3, 4]])

# 软标签 (每行是类别概率分布)
soft_labels = np.array([
    [0.7, 0.2, 0.1],   # 样本1: 类别0概率0.7
    [0.1, 0.8, 0.1],   # 样本2: 类别1概率0.8
    [0.3, 0.3, 0.4]    # 样本3: 类别2概率0.4
])

# 展开样本
X_expanded, y_expanded, w_expanded = [], [], []
for xi, yi in zip(X, soft_labels):
    for cls, prob in enumerate(yi):
        if prob > 0:  # 仅保留非零概率
            X_expanded.append(xi)
            y_expanded.append(cls)
            w_expanded.append(prob)
            print()
        print()
    print()

X_expanded = np.array(X_expanded)
y_expanded = np.array(y_expanded)
w_expanded = np.array(w_expanded)

# 用样本权重训练决策树
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_expanded, y_expanded, sample_weight=w_expanded)

# 预测 (返回类别概率分布)
print("预测类别:", clf.predict([[2.5, 3.5]]))
print("预测概率:", clf.predict_proba([[2.5, 3.5]]))
