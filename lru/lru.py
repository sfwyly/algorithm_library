

"""
    @Author: Junjie Jin
    @Date: 2022/4/25
    @Description: 基于双向链表与hash 实现 lru  添加
"""

from collections import defaultdict


class Node:
    def __int__(self, key=None, value=None):
        self.key = key
        self.value = value

        self.prev = None
        self.next = None

    def __eq__(self, other):
        return self.key == other.key and self.value == self.value and self.prev == other.prev and self.next == other.next

    def __hash__(self):
        return hash(self)


class DoubleList:

    def __init__(self):
        self.head = None
        self.tail = None
        self.capacity = 0

    # 在链表头部添加节点x
    def addFirst(self, node: Node):
        if not self.tail:
            self.head = self.tail = node
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
        self.capacity += 1
        return 1

    # 删除x节点一定存在
    def remove(self, node: Node):
        if self.capacity == 1:
            self.head = self.tail = None
        else:
            if node == self.head:
                next = node.next
                next.prev = None
                self.head = next
            elif node == self.tail:
                prev = node.prev
                prev.next = None
                self.tail = prev
            else:
                prev = node.prev
                prev.next = node.next
                node.next.prev = prev
        del node
        self.capacity -= 1
        return 1

    # 删除链表中最后一个节点，并返回该节点
    def removeLast(self):
        # if self.capacity == 1:
        #     del self.tail
        #     self.head = self.tail = None
        # else:
        #     prev = self.tail.prev
        #     prev.next = None
        #     del self.tail
        #     self.tail = prev
        self.remove(self.tail)
        return 1

    # 返回链表长度
    def size(self):
        return self.capacity


class LRUCache:

    def __int__(self):
        # key -> Node(key, val) hashmap
        self.map = defaultdict(Node)
        # Node <=> Node <=> Node  双向链表
        self.cache = DoubleList()
        # 最大容量
        self.cap = 10

    def get(self, key):
        if key in self.map:
            value = self.map[key].value
            # 引用传递 这么写会有问题的 节点已经被释放了 哪里还有数据添加
            # node = self.map[key]
            # self.cache.remove(node)
            # self.cache.addFirst(node)
            self.put(key, value)
            return value
        return -1

    def put(self, key, value):
        if key in self.map:
            self.cache.remove(self.map[key])
        elif self.cache.size() == self.cap:  # 删除最久未使用 key 也应该被删除
            del self.map[self.cache.tail.key]
            self.cache.removeLast()
        node = Node(key, value)
        self.map[key] = node
        self.cache.addFirst(node)
        return 1




