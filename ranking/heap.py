
"""
    实现堆排
"""

def rankRange(nums, l, r):

    tmp = nums[l]
    cur = l
    while cur < r:

        left = 2 * cur + 1
        right = 2 * cur + 2

        if left < r and right < r:

            if nums[left] <= nums[right]:
                m = right
            else:
                m = left
        elif left < r:
            m = left
        elif right < r:
            m = right
        else:
            break
        if nums[m] >= tmp:
            nums[cur] = nums[m]
            cur = m
        else:
            break
    nums[cur] = tmp


def rank(nums, r):

    for i in range(r//2, -1, -1):
        rankRange(nums, i, r)

    for i in range(r):

        tmp = nums[0]
        nums[0] = nums[r-i-1]
        nums[r-i-1] = tmp
        rankRange(nums, 0, r - i - 1)
    print(nums)


nums = [0, 8 ,7, 6, 5, 4, 3, 2, 1, 9]
rank(nums, len(nums))
