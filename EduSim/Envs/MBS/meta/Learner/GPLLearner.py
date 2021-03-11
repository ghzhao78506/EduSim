# coding: utf-8
# 2020/5/13 @ tongshiwei

from collections import namedtuple
from EduSim.Envs.MBS.utils import gpl
from EduSim.Envs.meta import MetaLearner, MetaLearningModel
from .MBSLearner import MBSLearnerGroup
import numpy as np
from copy import deepcopy

GPLState = namedtuple("GPLState", ["n_correct", "n_attempts", "latest_review_ts", "window_index"])


class GPLLearningModel(MetaLearningModel):
    def __init__(self, window_size, n_items):
        self._window_size = window_size
        self.n_items = n_items

    def step(self, state: GPLState, learning_item_id, correct, timestamp, *args, **kwargs):
        # 改动一:Tuple 元素不能直接修改
        # state.window_index += 1
        state = state._replace(window_index=state.window_index + 1)
        state = state._replace(n_correct=np.vstack([state.n_correct, [0] * self.n_items]))
        state = state._replace(n_attempts=np.vstack([state.n_attempts, [0] * self.n_items]))

        if correct:
            state.n_correct[-1][learning_item_id] += 1
        state.n_attempts[-1][learning_item_id] += 1
        state.latest_review_ts[learning_item_id] = timestamp


class GPLLearner(MetaLearner):
    def __init__(self, latest_review_ts: list, ability: list, ability_coefficient: list,
                 delay_coefficient: float, window_correct_coefficients: list = None,
                 window_attempt_coefficients: list = None,
                 window_size=40, n_items=0, n_windows=0):
        super(GPLLearner, self).__init__()

        # 改动二
        self._state = GPLState(
            np.zeros((window_size * n_windows, n_items)),
            np.zeros((window_size * n_windows, n_items)),
            latest_review_ts,
            0
        )

        self._learning_model = GPLLearningModel(window_size, len(latest_review_ts))
        self._ability = ability
        self._ability_coefficient = ability_coefficient
        self._decay_rate = delay_coefficient
        self._window_correct_coefficients = window_correct_coefficients
        self._window_attempt_coefficients = window_attempt_coefficients
        self._window_size = window_size
        self._n_windows = n_windows

    def response(self, test_item, timestamp, *args, **kwargs) -> ...:
        item_n_corrects = []
        item_n_attempts = []
        # shape 5*30
        cur_index = self._state.window_index
        for _ in range(self._n_windows):
            item_n_corrects.append(
                sum(self._state.n_correct[cur_index:cur_index + self._window_size])
            )
            item_n_attempts.append(
                sum(self._state.n_attempts[cur_index:cur_index + self._window_size])
            )
            cur_index += self._window_size

        item_n_attempts = np.asarray(item_n_attempts)
        item_n_corrects = np.asarray(item_n_corrects)

        # 改动三
        return gpl(
            self._ability[test_item.id],
            self._ability_coefficient[test_item.id],
            test_item.attribute["difficulty"],
            test_item.attribute["difficulty_coefficient"],
            timestamp - self._state.latest_review_ts[test_item.id],
            self._decay_rate,
            item_n_corrects[:, test_item.id],
            item_n_attempts[:, test_item.id],

            self._window_correct_coefficients,
            self._window_attempt_coefficients
        )

    def learn(self, learning_item, timestamp, *args, **kwargs):
        # TODO
        prob = self.response(test_item=learning_item, timestamp=timestamp)
        correct = True if np.random.random() < prob else False
        self._learning_model.step(self._state, learning_item_id=learning_item.id, correct=correct, timestamp=timestamp)

    @property
    def state(self):
        return self._state


class GPLLearnerGroup(MBSLearnerGroup):
    def __init__(self, n_items, n_steps=None, n_windows=5, seed=None):
        self.n_items = n_items
        self.n_windows = n_windows
        super(GPLLearnerGroup, self).__init__(seed=seed)
        self._window_correct_coefficients = self.sample_window_cw(n_windows)
        self._window_attempt_coefficients = self.sample_window_nw(n_windows)
        self._ability = self.sample_student_ability(self.n_items)
        self._ability_coefficient = self.sample_student_ability_coefficient(self.n_items)
        self._init_latest_review_ts = self.sample_init_review_time(self.n_items, sample_type="normal")
        self._delay_coefficient = self.sample_delay_coefficient()
        self._window_size = n_steps // n_windows if n_steps is not None else 40

    def __next__(self):
        return GPLLearner(
            latest_review_ts=deepcopy(self._init_latest_review_ts),
            ability=self._ability,
            ability_coefficient=self._ability_coefficient,
            delay_coefficient=self._delay_coefficient,
            window_correct_coefficients=self._window_correct_coefficients,
            window_attempt_coefficients=self._window_attempt_coefficients,
            window_size=self._window_size,
            n_items=self.n_items,
            n_windows=self.n_windows
        )
