class MultiLevelCache(object):
    def __init__(self):
        self._l1_key = None
        self._l1_value = None

        self._l2_size = 3
        self._l2 = [(None, None) for _ in range(self._l2_size)]
        self._l2_ptr = 0

        self._l3 = dict()

    def set(self, key, value):
        self._l1_key = key
        self._l1_value = value

        self._l2[self._l2_ptr] = (key, value)
        self._l2_ptr = (self._l2_ptr + 1) % 3  # l2_size: 3

        self._l3[key] = value

    def get(self, key, *args):
        if key == self._l1_key:
            return self._l1_value

        l2 = self._l2
        if key == l2[0][0]:
            return l2[0][1]

        if key == l2[1][0]:
            return l2[1][1]

        if key == l2[2][0]:
            return l2[2][1]

        return self._l3.get(key, *args)

    def containe(self, key):
        return key in self._l3

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, key, value):
        self.set(key, value)

    def __contains__(self, item):
        if item == self._l1_key:
            return True

        l2 = self._l2
        if item == l2[0][0] or item == l2[1][0] or item == l2[2][0]:
            return True

        return item in self._l3
