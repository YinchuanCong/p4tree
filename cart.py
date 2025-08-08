import numpy as np

class CARTNode:
    def __init__(self, feature=None, threshold=None, label=None, left=None, right=None):
        """
        决策树中的节点类
        :param feature: 用于分裂的特征索引
        :param threshold: 分裂阈值
        :param label: 如果是叶子节点，存储soft标签分布
        :param left: 左子树
        :param right: 右子树
        """
        self.feature = feature
        self.threshold = threshold
        self.label = label  # soft label 概率分布
        self.left = left
        self.right = right


class SoftCARTClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        支持soft标签的CART分类器
        :param max_depth: 树的最大深度
        :param min_samples_split: 最小划分样本数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """
        拟合训练数据
        :param X: 特征数组 (n_samples, n_features)
        :param y: soft标签 (n_samples, n_classes)
        """
        self.root = self._build_tree(X, y, depth=0)

    def _soft_gini(self, y_soft):
        """
        使用soft标签计算Gini指数
        :param y_soft: 概率标签 (n_samples, n_classes)
        :return: Gini值
        """
        probs = np.mean(y_soft, axis=0)
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        """
        寻找最佳划分特征与阈值
        :return: (feature, threshold)
        """
        best_gini, best_feature, best_thresh = float('inf'), None, None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for thresh in thresholds:
                left_mask = X[:, feature] <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                gini_left = self._soft_gini(y[left_mask])
                gini_right = self._soft_gini(y[right_mask])
                gini_split = (len(y[left_mask]) * gini_left + len(y[right_mask]) * gini_right) / len(y)
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feature = feature
                    best_thresh = thresh
        return best_feature, best_thresh

    def _build_tree(self, X, y, depth):
        """
        递归构建决策树
        """
        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            label_dist = np.mean(y, axis=0)
            return CARTNode(label=label_dist)

        feat, thresh = self._best_split(X, y)
        if feat is None:
            label_dist = np.mean(y, axis=0)
            return CARTNode(label=label_dist)

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return CARTNode(feature=feat, threshold=thresh, left=left, right=right)

    def predict_proba(self, X):
        """
        预测soft标签（每个样本的概率分布）
        :param X: (n_samples, n_features)
        :return: (n_samples, n_classes)
        """
        return np.array([self._predict_sample(x, self.root) for x in X])

    def predict(self, X):
        """
        预测硬标签（最大概率类别）
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def _predict_sample(self, x, node):
        """
        遍历树对单个样本预测
        """
        if node.label is not None:
            return node.label
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
        
    def _tree_torule(self):
        """
        打印决策树的规则路径，每条规则对应一条从根到叶的路径
        """
        def recurse(node, path):
            if node.label is not None:
                rule_str = "if " + " and ".join(path) if path else "Always"
                print(f"{rule_str} => {np.round(node.label, 4).tolist()}")
                return
            cond = f"feature_{node.feature} <= {node.threshold}"
            recurse(node.left, path + [cond])
            cond = f"feature_{node.feature} > {node.threshold}"
            recurse(node.right, path + [cond])

        recurse(self.root, [])
    
def tree_to_table(node, path=None):
    """
    遍历决策树，转为 match-action 表项
    :param node: CARTNode，当前节点
    :param path: 当前路径上的匹配条件，list of (feature_idx, op, threshold)
    :return: list of dict，包含 bits 和 action
    """
    if path is None:
        path = []

    if node.label is not None:
        # 到达叶子节点，构造bits条件
        feature_max = max((f for f, *_ in path), default=-1) + 1
        bits = ['*'] * feature_max
        for feat_idx, op, thresh in path:
            # 二值编码：<=阈值为0，>为1
            if op == '<=':
                bits[feat_idx] = '0'
            elif op == '>':
                bits[feat_idx] = '1'
        return [{'bits': bits, 'action': node.label.tolist()}]

    # 继续向左右子树递归
    left_path = path + [(node.feature, '<=', node.threshold)]
    right_path = path + [(node.feature, '>', node.threshold)]
    left_entries = tree_to_table(node.left, left_path)
    right_entries = tree_to_table(node.right, right_path)
    return left_entries + right_entries


def export_to_p4_table_entries(table, feature_prefix="hdr.meta.feature", action_name="set_label"):
    """
    将 match-action 表项导出为 P4 table 添加语句格式
    :param table: list of {'bits': [...], 'action': [...]}
    :return: list of P4 table entry strings
    """
    p4_entries = []
    for entry in table:
        bits = entry['bits']
        action = entry['action']
        match_fields = []
        for idx, b in enumerate(bits):
            field = f"{feature_prefix}_{idx}"
            if b == '*':
                continue  # 通配符时省略
            match_fields.append(f"{field} == {b}")
        match_str = " && ".join(match_fields) if match_fields else "true"
        action_str = f"{action_name}({', '.join(map(str, action))})"
        p4_entry = f'if ({match_str}) {{ {action_str}; }}'
        p4_entries.append(p4_entry)
    return p4_entries
