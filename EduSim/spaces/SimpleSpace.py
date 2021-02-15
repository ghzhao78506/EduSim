# coding: utf-8
# 2021/1/28 @ tongshiwei
from pprint import pformat
from gym.spaces import Space

__all__ = ["ListSpace"]


class ListSpace(Space):
    def __init__(self, elements: list, seed=None):
        self.elements = elements
        super(ListSpace, self).__init__(shape=(len(self.elements),))
        self.seed(seed)

    def sample(self):
        return self.np_random.choice(self.elements)

    def contains(self, item):
        return item in self.elements

    def __repr__(self):
        return pformat(self.elements)
