# coding: utf-8
# 2020/5/13 @ tongshiwei

from copy import deepcopy
import numpy as np
from EduSim.Envs.meta import Env
from EduSim.spaces import ListSpace
from EduSim.Envs.MBS.meta.Learner import EFCLearnerGroup, HLRLearnerGroup, GPLLearnerGroup
from .utils import sample
from .meta import Reward, MBSItemBase


class MetaEnv(Env):
    def __init__(self, n_items=30, reward_func="likelihood", threshold=0.5, seed=None, *args, **kwargs):
        self.n_items = n_items
        self.timestamp = 0
        self._learner = None
        self._reward = Reward(reward_func=reward_func)
        self._threshold = threshold
        self.random_state = np.random.RandomState(seed)
        self.learning_item_base: MBSItemBase = None
        self.test_item_base: MBSItemBase = None

    def sample_delay(self, sample_type="const", a=5, b=None):
        return sample(sample_type, a, b, random_state=self.random_state)

    def test(self, item, timestamp):
        raise NotImplementedError

    def exam(self):

        observation = [
            [
                test_item.id,
                1 if self.random_state.random() < self.test(test_item, self.timestamp) else 0,
                self.timestamp
            ] for test_item in self.test_item_base
        ]
        probabilities = [
            obs[1] for obs in observation
        ]
        reward = self._reward(probabilities)
        return observation, probabilities, reward

    def step(self, learning_item_id: int, *args, **kwargs):
        timestamp = self.timestamp

        self._learner.learn(self.learning_item_base[learning_item_id], timestamp)

        observation, probabilities, reward = self.exam()

        done = all([p > self._threshold for p in probabilities])
        info = {}

        self.timestamp += self.sample_delay(sample_type="const", a=5)
        return observation, reward, done, info

    def n_step(self, learning_path, *args, **kwargs):
        for learning_item_id in learning_path:
            self.step(learning_item_id)

    def end_episode(self, *args, **kwargs):
        observation, probabilities, reward = self.exam()
        done = all([p > self._threshold for p in probabilities])
        info = {}

        return observation, reward, done, info

    def reset(self):
        self._learner = None
        self.timestamp = 0

    def render(self, mode='human'):
        if mode == "log":
            return self._learner.state

    def begin_episode(self, *args, **kwargs):
        raise NotImplementedError


class EFCEnv(MetaEnv):
    def __init__(self, n_items=30, reward_func="likelihood", threshold=0.5, seed=None, *args, **kwargs):
        super(EFCEnv, self).__init__(
            n_items=n_items,
            reward_func=reward_func,
            threshold=threshold,
            seed=seed,
            *args, **kwargs
        )
        self.test_item_base = self._item_base = MBSItemBase(
            self.n_items,
            attributes={"difficulty"},
            seed=seed
        )
        self.learning_item_base = deepcopy(self._item_base).drop_attribute()
        self.learners = EFCLearnerGroup(self.n_items, seed=seed)
        self.action_space = ListSpace(self.learning_item_base.item_id_list, seed=seed)

    @property
    def parameters(self) -> dict:
        return {
            "action_space": self.action_space
        }

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)
        observation, *_ = self.exam()
        return self._learner.profile, observation

    def test(self, item, timestamp):
        return self._learner.response(item, timestamp)


class HLREnv(MetaEnv):
    def __init__(self, n_items=30, reward_func="likelihood", threshold=0.5, seed=None, *args, **kwargs):
        super(HLREnv, self).__init__(
            n_items=n_items,
            reward_func=reward_func,
            threshold=threshold,
            seed=seed,
            *args, **kwargs
        )
        self.learners = HLRLearnerGroup(self.n_items, seed=seed)
        self.test_item_base = self._item_base = MBSItemBase(self.n_items, seed=seed)
        self.learning_item_base = deepcopy(self._item_base)
        self.learning_item_base.drop_attribute()
        self.action_space = ListSpace(self.learning_item_base.item_id_list, seed=seed)

    @property
    def parameters(self) -> dict:
        return {
            "action_space": self.action_space
        }

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)
        observation, *_ = self.exam()
        return self._learner.profile, observation

    def test(self, item, timestamp):
        return self._learner.response(item, timestamp)


class GPLEnv(MetaEnv):
    def __init__(self, n_items=30, reward_func="likelihood", threshold=0.5, seed=None, n_steps=None, *args, **kwargs):
        super(GPLEnv, self).__init__(
            n_items=n_items,
            reward_func=reward_func,
            threshold=threshold,
            seed=seed,
            *args, **kwargs
        )
        self.learners = GPLLearnerGroup(self.n_items, n_steps=n_steps, seed=seed)
        self.test_item_base = self._item_base = MBSItemBase(
            self.n_items,
            attributes={"difficulty", "difficulty_coefficient"},
            seed=seed
        )
        self.learning_item_base = deepcopy(self._item_base)
        # self.learning_item_base.drop_attribute()
        self.test_item_base = self._item_base

        self.action_space = ListSpace(self.learning_item_base.item_id_list, seed=seed)

    @property
    def parameters(self) -> dict:
        return {
            "action_space": self.action_space
        }

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)
        observation, *_ = self.exam()
        return self._learner.profile, observation

    def test(self, item, timestamp):
        # print(item.id)
        return self._learner.response(item, timestamp)
