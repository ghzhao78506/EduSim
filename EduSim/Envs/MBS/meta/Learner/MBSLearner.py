# coding: utf-8
# 2020/5/13 @ tongshiwei

import numpy as np
from EduSim.Envs.meta import MetaInfinityLearnerGroup
from EduSim.Envs.MBS.utils import sample

__all__ = ["MBSLearnerGroup"]


class MBSLearnerGroup(MetaInfinityLearnerGroup):
    def __init__(self, seed=None):
        super(MBSLearnerGroup, self).__init__()
        self.random_state = np.random.RandomState(seed)

    def sample_memory_strengths(self, n_items, sample_type="const", a=1, b=None):
        return [sample(sample_type, a, b, random_state=self.random_state) for _ in range(n_items)]

    @classmethod
    def sample_log_linear_feature(cls, n_items):
        _features = np.zeros((n_items, 3))
        return np.concatenate((_features, np.eye(n_items)), axis=1).tolist()

    def sample_log_linear_feature_coefficients(self, n_items):
        return np.concatenate((np.array([1, 1, 0]), self.random_state.normal(0, 1, n_items))).tolist()

    def sample_init_review_time(self, n_items, sample_type="const"):
        if sample_type == "const":
            return [-np.inf] * n_items
        elif sample_type == "normal":
            return (-np.exp(self.random_state.normal(0, 1, n_items))).tolist()
        else:
            raise TypeError("unknown sample type: %s" % sample_type)

    @staticmethod
    def sample_student_ability(n_items):
        return [0] * n_items

    @staticmethod
    def sample_student_ability_coefficient(n_items):
        return [0] * n_items

    @staticmethod
    def sample_window_cw(n_windows):
        x = 1 / (np.arange(1, n_windows + 1, 1)) ** 2
        return x[::-1]

    @staticmethod
    def sample_window_nw(n_windows):
        x = 1 / (np.arange(1, n_windows + 1, 1)) ** 2
        return x[::-1]

    def sample_delay_coefficient(self):
        return np.exp(self.random_state.normal(0, 0.01))

    def __next__(self):
        raise NotImplementedError
