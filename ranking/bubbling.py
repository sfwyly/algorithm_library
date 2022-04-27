

"""
    冒泡排序
"""


def rank(nums):

    length = len(nums)

    for i in range(length):

        for j in range(length-1-i):

            if nums[j] > nums[j + 1]:

                tmp = nums[j]
                nums[j] = nums[j + 1]
                nums[j + 1] = tmp
    print(nums)

nums = [9,8,7,6,5,4,3,2,1,0]
rank(nums)