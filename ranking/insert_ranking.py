

"""
    插入排序
"""


def rank(nums):

    length = len(nums)

    for i in range(length):
        tmp = nums[i]
        j = i - 1
        while j >= 0 and nums[j] > tmp:
            nums[j+1] = nums[j]
            j -= 1

        nums[j+1] = tmp
    print(nums)


nums = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
rank(nums)
