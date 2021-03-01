# coding: utf-8
# 2021/2/18 @ tongshiwei

import numpy as np
from EduSim.Envs.meta import ItemBase
from networkx import Graph, DiGraph

__all__ = ["KSSItemBase"]


class KSSItemBase(ItemBase):
    def __init__(self, knowledge_structure: (Graph, DiGraph), learning_order=None, items=None, seed=None):
        self.random_state = np.random.RandomState(seed)
        if items is None:
            assert learning_order is not None
            difficulties = list(
                sorted([self.random_state.randint(0, 5) for _ in range(len(knowledge_structure.nodes))]))
            items = [
                {
                    "knowledge": node,
                    "attribute": {
                        "difficulty": difficulties[i]
                    }
                } for i, node in enumerate(knowledge_structure.nodes)
            ]

        super(KSSItemBase, self).__init__(
            items, knowledge_structure=knowledge_structure,
        )
        self.knowledge2item = dict()
        for item in self.items:
            self.knowledge2item[item.knowledge] = item
