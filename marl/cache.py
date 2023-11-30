from collections import deque


class Cache:
    def __init__(self, purpose='purpose not clear yet'):
        self.purpose = purpose
        self.cache = {}

    def __setitem__(self, key, value=deque(maxlen=int(1e4))):  # 增加或修改函数
        self.cache[key] = value

    def __getitem__(self, item):  # 获取函数
        return self.cache[item]

    def __delitem__(self, key):  # 删除函数
        del self.cache[key]
