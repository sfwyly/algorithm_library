
"""
    @Description: Kmp算法匹配
    @Date: 2021/11/22

"""


def getNext(p):
    length = len(p)
    next = [0 for _ in range(length)]
    i, j = 0, -1
    next[0] = -1
    while i < length-1:
        if j == -1 or p[i] == p[j]:
            i += 1
            j += 1
            next[i] = j
        else:
            j = next[j]
    return next


def kmp(s, p):
    next = getNext(p)
    ls, lp = len(s), len(p)

    i = 0
    j = 0

    while i < ls and j < lp:

        if j == -1 or s[i] == p[j]:
            i += 1
            j += 1
        else:
            j = next[j]
    if j == lp:
        return i - lp
    return -1


print(kmp("abcbcbsbbbcaabbabcba", "bbcaabb"))
