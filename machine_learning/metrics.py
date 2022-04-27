#
# """
#     @Author: junjie Jin
#     @Date: 2021/12/25
#     @Description: 计算各种指标
# """
import numpy as np
import matplotlib.pyplot as plt


class Metric:

    def __init__(self):
        pass

    def getMetrics(self, y_true, y_pred):
        """
        获取指标 输入numpy数组
        :param y_true: 0, 1
        :param y_pred: 0， 1
        :return:
        """
        assert len(y_true) == len(y_pred), "数组长度不相等"
        n = len(y_true)

        # TP FP TN FN

        TP = np.sum((y_true == y_pred) * y_pred)
        FP = np.sum((1 - (y_true == y_pred)) * y_pred)
        TN = np.sum((y_true == y_pred) * (1 - y_pred))
        FN = np.sum((1 - (y_true == y_pred)) * (1 - y_pred))
        # accuracy
        acc = (TP + TN) / (TP + TN + FP + FN)

        # precision
        prec = TP / (TP + FP)

        # recall
        recall = TP / (TP + FN)

        # TPR
        tpr = recall

        # FPR
        fpr = FP / (FP + TN)

        return {"acc": acc, "precision": prec, "recall": recall, "tpr": tpr, "fpr": fpr}

    def getROC(self, clses, scores):
        """
        根据真实标签与预测分数获取
        :param clses:
        :param scores:
        :return:
        """

        assert len(clses) == len(scores), "数组长度不相等"
        n = len(clses)
        nums = sorted(scores, reverse=True)
        res = []
        for i in range(n):
            dic = self.getMetrics(clses, scores >= nums[i])
            res.append([dic["fpr"], dic["tpr"]])
        return np.array(res)

    def getAUC(self, clses, scores):
        nums = []
        n = len(clses)
        for _, (cls, score) in enumerate(zip(clses, scores)):
            nums.append([cls, score])
        nums.sort(key=lambda x:x[1], reverse=True)
        M, N = 0, 0
        rank = 0
        for i, (cls, score) in enumerate(nums):
            if cls == 1:
                M += 1
                rank += (n - i)
            else:
                N += 1
        auc = (rank - M* (1 + M) // 2) / (M*N)
        return auc


if __name__ == '__main__':
    metric = Metric()
    clses = [1, 1, 0, 1,1,1,0,0,1,0, 1,0,1,0,0,0,1,0,1,0]
    scores = [0.9, 0.8,0.7,0.6,0.55,0.54,0.53,0.52,0.51,0.505, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.30, 0.1]
    res = metric.getROC(np.array(clses), np.array(scores))
    plt.plot(res[:, 0], res[:, 1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    print(metric.getAUC(clses, scores))
    plt.show()
    print(0.02 + 0.1 + 0.06 + 0.07 + 0.24 + 0.09 + 0.1)

# print(Metric().getMetrics(np.array([1,1,1,1,1,0,0,0,0,0]),np.array([1,1,1,1,1,0,0,0,0,0])))
