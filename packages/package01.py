
"""
    @Description： 0 1 背包问题
"""


def solution(w, v, n, V):
    """
    :param w: length nums
    :param v: length nums
    :param n: all num things
    :param V: package max vector
    :return:
    """
    length = len(w)
    nums = [[0] * (V + 1) for _ in range(n)]

    return 0