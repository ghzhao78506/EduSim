# coding: utf-8
# 2020/5/13 @ tongshiwei

import numpy as np
from EduSim.Envs.meta import ItemBase


class MBSItemBase(ItemBase):
    def __init__(self, n_items, attributes=None, seed=None):
        self.n_items = n_items
        self.random_state = np.random.RandomState(seed)
        items = {}
        for i in range(n_items):
            items[i] = {}

        attributes = attributes if attributes is not None else {}

        difficulties = None if "difficulty" not in attributes else self.sample_item_difficulties()

        difficulty_coefficients = None if "difficulty_coefficient" not in attributes \
            else self.sample_difficulty_coefficients()

        for i in range(n_items):
            items[i]["attribute"] = {}
            if difficulties is not None:
                items[i]["attribute"]["difficulty"] = difficulties[i]
            if difficulty_coefficients is not None:
                items[i]["attribute"]["difficulty_coefficient"] = difficulty_coefficients[i]
        super(MBSItemBase, self).__init__(items)

    def sample_item_difficulties(self):
        return np.exp(self.random_state.normal(np.log(0.077), 1, self.n_items)).tolist()

    def sample_difficulty_coefficients(self):
        return np.exp(self.random_state.normal(1, 1, self.n_items))
