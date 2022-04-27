

def leftChild(index):
    return index * 2 + 1


def rightChild(index):
    return index * 2 + 2


class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.data = arr[:]

    def getSize(self):
        return self.n

    def get(self, index):
        return self.data[index]

    def buildSegmentTree(self, treeIndex, l, r):
        if l == r:
            self.tree[treeIndex] = self.data[l]
            return
        leftTreeIndex = leftChild(treeIndex)
        rightTreeIndex = rightChild(treeIndex)
        mid = l + (r - l) // 2
        self.buildSegmentTree(leftTreeIndex, l, mid)
        self.buildSegmentTree(rightTreeIndex, mid + 1, r)
        self.tree[treeIndex] = self.merge(self.tree[leftTreeIndex], self.tree[rightTreeIndex])

    def merge(self, a, b):
        return max(a, b)

    def query(self, treeIndex, l, r, queryL, queryR):
        if l == queryL and r == queryR:
            return self.tree[treeIndex]
        mid = l + (r - l) // 2
        leftTreeIndex = leftChild(treeIndex)
        rightTreeIndex = rightChild(treeIndex)
        if queryL > mid:
            return self.query(rightTreeIndex, mid + 1, r, queryL, queryR)

        if queryR <= mid:
            return self.query(leftTreeIndex, l, mid, queryL, queryR)

        leftResult = self.query(leftTreeIndex, l, mid, queryL, mid)
        rightResult = self.query(rightTreeIndex, mid + 1, r, mid + 1, queryR)

        return self.merge(leftResult, rightResult)

    def update(self, index, v):
        self.data[index] = v
        self.updateTree(0, 0, self.n - 1, index, v)

    def updateTree(self, treeIndex, l, r, index, v):
        if l == r:
            self.tree[treeIndex] = v
            return
        mid = l + (r - l) // 2
        leftTreeIndex = leftChild(treeIndex)
        rightTreeIndex = rightChild(treeIndex)

        if index > mid:
            self.updateTree(rightTreeIndex, mid + 1, r, index, v)
        else:
            self.updateTree(leftTreeIndex, l, mid, index, v)
        self.tree[treeIndex] = self.merge(self.tree[leftTreeIndex], self.tree[rightTreeIndex])
