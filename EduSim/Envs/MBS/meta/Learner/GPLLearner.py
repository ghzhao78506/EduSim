# coding: utf-8
# 2020/5/13 @ tongshiwei

from collections import namedtuple
from EduSim.Envs.MBS.utils import gpl
from EduSim.Envs.meta import MetaLearner, MetaLearningModel
from .MBSLearner import MBSLearnerGroup

GPLState = namedtuple("GPLState", ["n_correct", "n_attempts", "latest_review_ts", "window_index"])


class GPLLearningModel(MetaLearningModel):
    def __init__(self, window_size, n_items):
        self._window_size = window_size
        self.n_items = n_items

    def step(self, state: GPLState, learning_item_id, correct, timestamp, *args, **kwargs):
        state.window_index += 1
        if state.window_index // self._window_size >= len(state.n_correct):
            state.n_correct.append([0] * self.n_items)
        if correct:
            state.n_correct[-1][learning_item_id] += 1
        state.n_attempts[-1][learning_item_id] += 1
        state.latest_review_ts[learning_item_id] = timestamp


class GPLLearner(MetaLearner):
    def __init__(self, latest_review_ts: list, ability: list, ability_coefficient: list,
                 delay_coefficient: float, window_correct_coefficients: list = None,
                 window_attempt_coefficients: list = None,
                 window_size=40):
        super(GPLLearner, self).__init__()
        self._state = GPLState(
            [[0] * len(latest_review_ts)],
            [[0] * len(latest_review_ts)],
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

    def response(self, test_item, timestamp, *args, **kwargs) -> ...:
        return gpl(
            self._ability[test_item.id],
            self._ability_coefficient[test_item.id],
            test_item.attribute["difficulty"],
            test_item.attribute["difficulty_coefficient"],
            timestamp - self._state.latest_review_ts[test_item.id],
            self._decay_rate,
            self._state.n_correct[self._state.window_index // self._window_size][test_item.id],
            self._state.n_attempts[self._state.window_index // self._window_size][test_item.id],
            self._window_correct_coefficients,
            self._window_attempt_coefficients
        )

    def learn(self, learning_item, timestamp, *args, **kwargs):
        self.response(learning_item, timestamp)

    @property
    def state(self):
        return self._state


class GPLLearnerGroup(MBSLearnerGroup):
    def __init__(self, n_items, n_steps=None, n_windows=5, seed=None):
        self.n_items = n_items
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
            latest_review_ts=self._init_latest_review_ts,
            ability=self._ability,
            ability_coefficient=self._ability_coefficient,
            delay_coefficient=self._delay_coefficient,
            window_correct_coefficients=self._window_correct_coefficients,
            window_attempt_coefficients=self._window_attempt_coefficients,
            window_size=self._window_size
        )
