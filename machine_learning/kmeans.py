
"""
    @Author: Junjie Jin
    @Date: 2022/5/19
    @Description: Kmeans
"""

import numpy as np


def kmeans(nums, clusters):
    # nums : 二维数组 center: 聚类中心个数
    np.random.shuffle(nums)
    centers_ = nums[:clusters]
    print(centers_)
    i = 0
    while i < 10:
        dist = []
        for num in nums:
            dist.append((np.sum((num - centers_) ** 2, axis=-1) ** 0.5))
        new_centers_ = []
        for cluster in range(clusters):
            new_centers_.append((np.mean(nums[np.argmin(dist, axis=-1)==cluster], axis=0)))
        centers_ = np.array(new_centers_)
        print(centers_)
        i += 1

nums = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[4,6],[2,3]], dtype=np.float64)
kmeans(nums, 3)
# print(np.random.choice(range(10), 5))