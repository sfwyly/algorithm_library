
"""
    @Author: Junjie Jin
    @Date: 2022/6/13
    @Description: 计数排序
"""


def count_sort(nums):
    m = max(nums)
    ns = [0 for _ in range(m + 1)]
    for i, ni in enumerate(nums):
        ns[ni] += 1
    idx, ans = 0, 0
    for i in range(1, m + 1):
        j = 1
        while j <= ns[i]:
            if i == nums[idx]:
                ans += 1
            idx += 1
            j += 1
    return ns, ans

if __name__ == '__main__':
    print(count_sort([5, 4, 3, 1, 3, 2]))
