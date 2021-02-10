# coding: utf-8
# 2021/2/7 @ tongshiwei
import numpy as np
from EduSim.Envs.meta import ItemBase

__all__ = ["TMSTestItemBase", "TMSLearningItemBase"]


class TMSLearningItemBase(ItemBase):
    pass


class TMSTestItemBase(ItemBase):
    def __init__(self, skill_num, exercise_for_each_skill, seed=None):
        items = []
        self.random_state = np.random.RandomState(seed)
        for skill in range(skill_num):
            for _ in range(exercise_for_each_skill):
                items.append({
                    "knowledge": skill,
                    "attribute": {
                        "guessing": self.random_state.uniform(0.1, 0.3),
                        "slipping": self.random_state.uniform(0.1, 0.3),
                    }
                })

        super(TMSTestItemBase, self).__init__(items)
