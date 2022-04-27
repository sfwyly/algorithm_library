

"""
    实现快排
"""


def rank(nums, l, r):

    if l >= r:
        return -1

    tmp = nums[l]
    while l < r:

        while l < r and nums[r] >= tmp:
            r -= 1
        nums[l] = nums[r]
        while l < r and nums[l] <= tmp:
            l += 1
        nums[r] = nums[l]
    nums[l] = tmp
    return l


def partition(nums, l, r):

    mid = rank(nums, l, r)
    if mid == -1:
        return
    partition(nums, l, mid - 1)
    partition(nums, mid + 1, r)

nums = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
partition(nums, 0, len(nums)-1)
print(nums)


