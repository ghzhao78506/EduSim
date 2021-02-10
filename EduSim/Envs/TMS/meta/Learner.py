# coding: utf-8
# 2020/4/29 @ tongshiwei

import numpy as np
from longling.ML.utils import choice

from EduSim.Envs.meta import MetaInfinityLearnerGroup, MetaLearner, MetaLearningModel, Item


class TransitionMatrix(MetaLearningModel):
    def __init__(self, transition_matrix, state2vector):
        self.matrix = transition_matrix
        self._state2vector = state2vector

    def step(self, state, learning_item: Item):
        return choice(self.matrix[learning_item.knowledge][state])

    def state2vector(self, state):
        return self._state2vector[state]


class Learner(MetaLearner):
    def __init__(self, transition_matrix, state2vector, initial_state: int, _id=None):
        super(Learner, self).__init__(_id)
        self.learning_model = TransitionMatrix(transition_matrix, state2vector)
        self._state = initial_state

    @property
    def state(self):
        return self.learning_model.state2vector(self._state)

    def learn(self, learning_item, *args, **kwargs):
        self._state = self.learning_model.step(self._state, learning_item)

    def response(self, test_item: Item, *args, **kwargs) -> ...:
        return self.state[test_item.knowledge]


class LearnerGroup(MetaInfinityLearnerGroup):
    def __init__(self, transition_matrix, state2vector, initial_states: list, seed=None):
        self._random_state = np.random.RandomState(seed)
        self.transition_matrix = transition_matrix
        self.state2vector = state2vector
        self.initial_states = initial_states

    def __next__(self):
        return Learner(
            transition_matrix=self.transition_matrix,
            state2vector=self.state2vector,
            initial_state=self._random_state.choice(self.initial_states)
        )
