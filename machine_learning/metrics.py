#
# """
#     @Author: junjie Jin
#     @Date: 2021/12/25
#     @Description: 计算各种指标
# """
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from models.inception import InceptionV3
import torch
from scipy import linalg
import lpips


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

    def getPSNR(self, img_a, img_b):
        """
        计算峰值信噪比
        :param img_a:
        :param img_b:
        :return:
        """
        assert img_a.shape == img_b.shape, "图像尺寸大小不相同"
        height, width, _ = img_a.shape

        channel_mse = np.sum((img_a - img_b) ** 2, axis=(0, 1)) / (height * width)
        mse = np.mean(channel_mse)
        Max = 1.0  # 最大图像
        PSNR = 10. * np.log10(Max ** 2 / mse)  # 峰值信噪比
        return PSNR

    def getSSIM(self, img_a, img_b):
        """
        计算结构相似性
        :param img_a:
        :param img_b:
        :return:
        """
        return ssim(img_b, img_a, data_range=1, multichannel=True)

    def getFID(self, img_a, img_b):
        """
        获取
        :param img_a:
        :param img_b:
        :return:
        """
        def frechet_distance(mu, cov, mu2, cov2):
            cc, _ = np.linalg.sqrtm(np.dot(cov, cov2), disp=False)
            dist = np.sum((mu - mu2) ** 2) + np.trace(cov + cov2 - 2 * cc)
            return np.real(dist)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        inception = InceptionV3().eval().to(device)
        mu, cov = [], []
        for img in [img_a, img_b]:
            actvs = []
            actv = inception(torch.from_numpy(np.transpose(img[np.newaxis, ...], (0, 2, 3, 1))))
            actvs.append(actv)
            actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
            mu.append(np.mean(actv, axis=0))
            cov.append(np.cov(actvs, rowvar=False))
        fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
        return fid_value

    def getLPIPS(self, net, img_a, img_b):
        """
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
        :param net:
        :param img_a:
        :param img_b:
        :return:
        """
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        loss_fn = lpips.LPIPS(net=net).to(device)
        img0 = lpips.im2tensor(img_a)
        img1 = lpips.im2tensor(img_b)
        return loss_fn.forward(img0, img1)


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
