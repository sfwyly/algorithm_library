
"""
    @Date: 2021/11/20
    @Description： 二分搜索
"""


def binary_search(nums, target):

    l, r = 0, len(nums) - 1

    while l < r:

        mid = (l + r) // 2
        if nums[mid] < target:
            l = mid + 1
        elif nums[mid] >= target:
            r = mid
    return l

nums = [0, 1, 2, 3, 4,5,6,7,8,9]
print(binary_search(nums, 9))