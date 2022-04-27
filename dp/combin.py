
"""
    @Date: 2021/11/17
    给定一个目标数值 与一个货币数组 ， 每种货币能够用多次，有多少种不同的组成方法
    组合数
"""


def solution(coins, amount):

    dp = [0 for _ in range(amount + 1)]
    dp[0] = 1

    for coin in coins:
        for i in range(amount+1):

            if i >= coin:
                dp[i] += dp[i - coin]
    return dp[amount]


"""
    最短能够成的目标的数字个数
"""


def solution(coins, amount):

    dp = [amount for _ in range(amount)]

    for i in range(amount + 1):

        for coin in coins:

            if i >= coin:

                dp[i] = min(dp[i], dp[i - coin] + 1)
    return amount
