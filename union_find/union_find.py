
"""
    @Date: 2021/12/28
    @Description:
    并查集
"""


parents = []


def find(i):
    if i == parents[i]:
        return i
    parents[i] = find(parents[i])
    return parents[i]


def union(i, j):
    parents[find(i)] = find(j)
