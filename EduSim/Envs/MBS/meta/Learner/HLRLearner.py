# coding: utf-8
# 2020/5/13 @ tongshiwei
from copy import deepcopy
import numpy as np
from collections import namedtuple
from EduSim.Envs.MBS.utils import hlr
from EduSim.Envs.meta import MetaLearner, MetaLearningModel
from .MBSLearner import MBSLearnerGroup

HLRState = namedtuple("HLRState", ["features", "latest_review_ts"])


class HLRLearningModel(MetaLearningModel):
    def step(self, state: HLRState, learning_item_id, outcome, timestamp, *args, **kwargs):
        state.features[learning_item_id][0] += 1
        state.features[learning_item_id][1 if outcome == 1 else 2] += 1
        state.latest_review_ts[learning_item_id] = timestamp


class HLRLearner(MetaLearner):
    def __init__(self, features, latest_review_ts, feature_coefficients):
        super(HLRLearner, self).__init__()
        self._state = HLRState(features, latest_review_ts)
        self._learning_model = HLRLearningModel()
        self._feature_coefficients = feature_coefficients

    @property
    def state(self):
        return HLRState([f[:3] for f in self._state.features], self._state.latest_review_ts)

    def learn(self, learning_item, timestamp, *args, **kwargs):
        self._learning_model.step(self._state, learning_item.id, self.response(learning_item, timestamp), timestamp)

    def response(self, test_item, timestamp, *args, **kwargs) -> ...:
        return hlr(
            timestamp - self._state.latest_review_ts[test_item.id],
            self._feature_coefficients,
            self._state.features[test_item.id],
        )


class HLRLearnerGroup(MBSLearnerGroup):
    def __init__(self, n_items, seed=None):
        self.n_items = n_items
        super(HLRLearnerGroup, self).__init__(seed)
        # n_attempts, n_correct, n_incorrect
        _features = np.zeros((self.n_items, 3))
        self._log_linear_features = self.sample_log_linear_feature(self.n_items)
        self._log_linear_features_coefficients = self.sample_log_linear_feature_coefficients(self.n_items)
        self._init_latest_review_ts = self.sample_init_review_time(self.n_items, sample_type="normal")

    def __next__(self):
        return HLRLearner(
            features=self._log_linear_features,
            latest_review_ts=deepcopy(self._init_latest_review_ts),
            feature_coefficients=self._log_linear_features_coefficients
        )
