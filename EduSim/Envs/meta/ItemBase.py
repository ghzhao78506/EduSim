# coding: utf-8
# 2020/4/29 @ tongshiwei

from collections import Iterable
from networkx import Graph, DiGraph
import uuid

__all__ = ["Item", "ItemBase"]


class Item(object):
    def __init__(self, knowledge=None, content: dict = None, attribute: dict = None, item_id=None):
        """attribute includes difficulty, trait and so on"""
        self.id = self.__id(item_id)
        self.knowledge = knowledge
        self._content = content
        self.attribute = attribute

    @classmethod
    def __id(cls, _id=None):
        return _id if _id is not None else uuid.uuid1()

    @property
    def content(self) -> (str, dict):
        return self._content if self._content is not None else {}

    @content.setter
    def content(self, value):
        self._content = value

    def update_content(self, value):
        self._content.update(value)
        return self.content

    def update_knowledge(self, value):
        self.knowledge = value
        return self.knowledge

    def __repr__(self):
        return str({"id": self.id, "content": self.content, "knowledge": self.knowledge})


def initial_item_base(items: (dict, list, int, Iterable)) -> list:
    """

    Parameters
    ----------
    items: dict, int, list of dict

    Returns
    -------
    item_base: list

    """
    if isinstance(items, int):
        items = [Item(knowledge=i) for i in range(items)]

    elif isinstance(items, dict):
        items = [Item(item_id=k, **v) for k, v in items.items()]

    elif isinstance(items, (list, Iterable)):
        items = [Item(**item) for item in items]

    else:
        raise NotImplemented("can not handle the type of %s" % type(items))

    return items


class ItemBase(object):
    def __init__(self,
                 items: (dict, list, int),
                 knowledge: (list, dict) = None,
                 knowledge_structure: (Graph, DiGraph) = None
                 ):
        self.items = initial_item_base(items)
        self.knowledge = knowledge
        self.knowledge_structure = knowledge_structure
        self.index = {
            item.id: item
            for item in self.items
        }

    def __getitem__(self, item):
        return self.index[item]

    def __contains__(self, item):
        return item in self.index

    def __iter__(self):
        return iter(self.items)

    @property
    def item_id_list(self):
        return list(self.index.keys())