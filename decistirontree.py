import numpy as np

class SoftLabelDecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def entropy(self, Y_soft):
        """计算软标签熵"""
        return -np.mean(np.sum(Y_soft * np.log(Y_soft + 1e-12), axis=1))

    def best_split(self, X, Y_soft):
        """寻找最佳划分特征"""
        n_samples, n_features = X.shape
        base_entropy = self.entropy(Y_soft)

        best_gain = 0
        best_feature = None
        best_splits = None

        for f in range(n_features):
            values = np.unique(X[:, f])
            splits = []
            split_entropy = 0

            for v in values:
                idx = X[:, f] == v
                if np.sum(idx) == 0:
                    continue
                splits.append((v, idx))
                split_entropy += (np.sum(idx) / n_samples) * self.entropy(Y_soft[idx])

            gain = base_entropy - split_entropy
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_splits = splits

        return best_feature, best_splits

    def build_tree(self, X, Y_soft, depth=0):
        """递归建树"""
        n_samples, n_features = X.shape

        # 叶子条件
        if (depth >= self.max_depth) or (n_samples < self.min_samples_split):
            return {"type": "leaf", "distribution": Y_soft.mean(axis=0)}

        # 找最佳划分
        feature, splits = self.best_split(X, Y_soft)
        if feature is None:
            return {"type": "leaf", "distribution": Y_soft.mean(axis=0)}

        node = {"type": "node", "feature": feature, "children": {}}
        for value, idx in splits:
            node["children"][value] = self.build_tree(X[idx], Y_soft[idx], depth + 1)
        return node

    def fit(self, X, Y_soft):
        self.tree = self.build_tree(X, Y_soft)

    def predict_one(self, x, node):
        if node["type"] == "leaf":
            return node["distribution"]
        feature = node["feature"]
        value = x[feature]
        if value in node["children"]:
            return self.predict_one(x, node["children"][value])
        else:
            return node["distribution"]  # 未见过的值，返回当前节点分布

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])


# # 特征：离散取值
# X = np.array([
#     [0, 1],
#     [0, 0],
#     [1, 1],
#     [1, 0]
# ])

# # 软标签（2分类）
# Y_soft = np.array([
#     [0.9, 0.1],
#     [0.8, 0.2],
#     [0.2, 0.8],
#     [0.1, 0.9]
# ])

# tree = SoftLabelDecisionTree(max_depth=2)
# tree.fit(X, Y_soft)

# print("预测分布：")
# print(tree.predict(np.array([[0, 1], [1, 0]])))
