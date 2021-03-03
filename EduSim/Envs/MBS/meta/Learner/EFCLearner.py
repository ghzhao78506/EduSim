# coding: utf-8
# 2020/5/13 @ tongshiwei

from copy import deepcopy
from collections import namedtuple
from EduSim.Envs.MBS.utils import efc
from EduSim.Envs.meta import MetaLearner, MetaLearningModel
from .MBSLearner import MBSLearnerGroup

EFCState = namedtuple("EFCState", ["memory_strengths", "latest_review_ts"])


class EFCLearningModel(MetaLearningModel):
    def step(self, state: EFCState, learning_item_id, timestamp, *args, **kwargs):
        state.memory_strengths[learning_item_id] += 1
        state.latest_review_ts[learning_item_id] = timestamp


class EFCLearner(MetaLearner):
    def __init__(self, memory_strengths: list, latest_review_ts: list):
        super(EFCLearner, self).__init__()
        self._state = EFCState(memory_strengths, latest_review_ts)
        self._learning_model = EFCLearningModel()

    @property
    def state(self):
        return self._state

    def learn(self, learning_item, timestamp, *args, **kwargs):
        self._learning_model.step(self._state, learning_item.id, timestamp)

    def response(self, test_item, timestamp, *args, **kwargs) -> ...:
        return efc(
            test_item.attribute["difficulty"],
            timestamp - self._state.latest_review_ts[test_item.id],
            self._state.memory_strengths[test_item.id]
        )


class EFCLearnerGroup(MBSLearnerGroup):
    def __init__(self, n_items, seed=None):
        self.n_items = n_items
        super(EFCLearnerGroup, self).__init__(seed)
        self._init_latest_review_ts = self.sample_init_review_time(self.n_items, sample_type="normal")

    def __next__(self) -> EFCLearner:
        return EFCLearner(
            memory_strengths=self.sample_memory_strengths(self.n_items),
            latest_review_ts=deepcopy(self._init_latest_review_ts)
        )
