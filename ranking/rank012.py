"""
    只有 0 1 2 三种数字的数组 O(n) 排序成012

"""


def rank(nums):

    length = len(nums)
    l = 0

    while l < length and nums[l] == 0:
        l += 1
    k = l

    r = length - 1

    while r >=0 and nums[r] == 2:
        r -= 1

    while k <= r:

        if nums[k] == 0:
            tmp = nums[k]
            nums[k] = nums[l]
            nums[l] = tmp

            while l<r and nums[l]==0:
                l += 1
            k = l
        elif nums[k] == 1:
            k +=1
        else:
            tmp = nums[k]
            nums[k] = nums[r]
            nums[r] = tmp

            while l<r and nums[r] ==2:
                r -= 1
    return nums


nums = [2,2,2,1,0,1,0,0,1,2,2,0,1]
print(rank(nums))