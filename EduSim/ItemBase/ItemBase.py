# coding: utf-8
# 2021/1/28 @ tongshiwei

from networkx import Graph, DiGraph


class ItemBase(object):
    """
    >>> item_base = ItemBase({1: "1", 2: "2"})
    >>> item_base.item_ids()
    [1, 2]
    """
    def __init__(self, items: (dict, set, list, tuple, int), knowledge: (dict, None) = None,
                 knowledge_structure: (Graph, DiGraph) = None):
        if isinstance(items, int):
            items = list(range(items))
        self.items = items
        self.knowledge = knowledge
        self.knowledge_structure = knowledge_structure

    def item_ids(self) -> (list, tuple, set):
        if isinstance(self.items, dict):
            return list(self.items.keys())
        else:
            return self.items
