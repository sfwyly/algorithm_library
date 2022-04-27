
"""
    @Date: 2021/12/28
    @Description:
    字典树 可用于字符串前缀匹配等操作
"""


class TrieNode:
    def __init__(self, val=None, is_end=False):
        self.val = val
        self.is_end = is_end
        self.children = dict()


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):

        cur = self.root
        for wi in word:
            if wi in cur.children:
                cur = cur.children[wi]
            else:
                cur.children[wi] = TrieNode(wi)
                cur = cur.children[wi]
        cur.is_end = True

    def search(self, word):
        cur = self.root
        for wi in word:
            if wi in cur.children:
                cur = cur.children[wi]
                continue
            return False
        if cur.is_end:
            return True
        return False
