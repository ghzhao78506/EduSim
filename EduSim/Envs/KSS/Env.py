# coding: utf-8
# 2020/4/30 @ tongshiwei

import networkx as nx
import random
from EduSim.Envs.meta import Env

import itertools
import numpy as np
from EduSim.Envs.KSS.meta.Learner import LearnerGroup, Learner
from EduSim.Envs.shared.KSS_KES import episode_reward, kss_kes_train_eval as kss_train_eval
from EduSim.spaces import ListSpace
from .meta import KSSItemBase, KSSScorer
from .utils import load_environment_parameters

__all__ = ["KSSEnv", "kss_train_eval"]

RANDOM = 0
LOOP = 1
INF = 2

MODE = {
    "random": RANDOM,
    "loop": LOOP,
    "inf": INF,
}


class KSSEnv(Env):
    def __init__(self, learner_num=4000, seed=None, initial_step=20):
        self.random_state = np.random.RandomState(seed)

        parameters = load_environment_parameters()
        self.knowledge_structure = parameters["knowledge_structure"]
        self._item_base = KSSItemBase(
            parameters["knowledge_structure"],
            parameters["learning_order"]
        )
        self.learning_item_base = self._item_base
        self.test_item_base = self._item_base
        self.scorer = KSSScorer()

        self.action_space = ListSpace(self.learning_item_base.items, seed=seed)

        self._learner_num = learner_num
        self.learners = LearnerGroup(self.knowledge_structure, seed=seed)

        self._order_ratio = parameters["configuration"]["order_ratio"]
        self._review_times = parameters["configuration"]["review_times"]
        self._learning_order = parameters["learning_order"]

        self._topo_order = list(nx.topological_sort(self.knowledge_structure))
        self._initial_step = initial_step

        self._learner = None

    @property
    def parameters(self) -> dict:
        return {
            "knowledge_structures": self.knowledge_structure,
            "action_space": self.action_space,
        }

    def _initial_logs(self, learner: Learner):
        logs = []

        if random.random() < self._order_ratio:
            while len(logs) < self._initial_step:
                if logs and logs[-1][1] == 1 and len(
                        set([e[0] for e in logs[-3:]])) > 1:
                    for _ in range(self._review_times):
                        if len(logs) < self._initial_step - self._review_times:
                            learning_item_id = test_item_id = logs[-1][0]
                            learner.learn(self.learning_item_base[str(learning_item_id)])
                            test_item = self.test_item_base[str(test_item_id)]
                            score = self.scorer(
                                learner.response(test_item), test_item.attributes
                            )
                            logs.append([test_item_id, score])
                        else:
                            break
                    learning_item_id = logs[-1][0]
                elif logs and logs[-1][1] == 0 and random.random() < 0.7:
                    learning_item_id = logs[-1][0]
                elif random.random() < 0.9:
                    for test_item_id in self._topo_order:
                        if learner.response(self.test_item_base[str(test_item_id)]) < 0.6:
                            break
                    else:  # pragma: no cover
                        break
                    learning_item_id = test_item_id
                else:
                    learning_item_id = self.random_state.choice(self.learning_item_base.index)

                learner.learn(self.learning_item_base[learning_item_id])
                test_item_id = learning_item_id
                test_item = self.test_item_base[str(test_item_id)]
                score = self.scorer(learner.response(test_item.knowledge), test_item.attributes)
                logs.append([test_item_id, score])
        else:
            while len(logs) < self._initial_step:
                if random.random() < 0.9:
                    for test_item_id in self._learning_order:
                        if learner.state[self.test_item_base[test_item_id].knowledge] < 0.6:
                            break
                    else:
                        break
                    learning_item_id = test_item_id
                else:
                    learning_item_id = self.random_state.choice(self.learning_item_base.index)

                learning_item = self.learning_item_base[learning_item_id]
                learner.learn(learning_item)
                test_item_id = learning_item_id
                test_item = self.test_item_base[test_item_id]
                score = self.scorer(learner.response(test_item), test_item.attributes)
                logs.append([test_item_id, score])

        learner.update_logs(logs)

    def _exam(self, learner: Learner, detailed=False, reduce="sum") -> (dict, int, float):
        knowledge_response = {}
        for test_knowledge in learner.target:
            knowledge_response[test_knowledge] = [
                [item.id, self.scorer(learner.response(item), item.attribute)]
                for item in self.test_item_base[test_knowledge]
            ]
        if detailed:
            return knowledge_response
        if reduce is None:
            return {k: np.average(v) for k, v in knowledge_response.items()}
        elif reduce == "sum":
            return np.sum([np.average(v) for v in knowledge_response.values()])
        elif reduce in {"mean", "ave"}:
            return np.average([np.average(v) for v in knowledge_response.values()])
        else:
            raise TypeError("unknown reduce type %s" % reduce)

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)
        self._initial_logs(self._learner)
        return self._learner.profile

    def end_episode(self, *args, **kwargs):
        observation = self._exerciser.exam(self._learner, *self._learner.target)
        initial_score = sum([v for _, v in self._exerciser.exam(self._initial_learner_state, *self._learner.target)])
        final_score = sum([v for _, v in observation])
        reward = episode_reward(initial_score, final_score, len(self._learner.target))
        done = final_score == len(self._learner.target)
        info = {"initial_score": initial_score, "final_score": final_score}

        assert reward >= 0, "%s" % self._idx

        return observation, reward, done, info

    def step(self, learning_item_id, *args, **kwargs):
        a = sum([v for _, v in self._exerciser.exam(self._learner, *self._learner.target)])
        self._learner.learn(learning_item_id)
        observation = self._exerciser.test(learning_item_id, self._learner)
        b = sum([v for _, v in self._exerciser.exam(self._learner, *self._learner.target)])

        return observation, b - a, b == len(self._learner.target), None

    def n_step(self, learning_path, *args, **kwargs):
        exercise_history = []
        a = sum([v for _, v in self._exerciser.exam(self._learner, *self._learner.target)])
        for learning_item_id in learning_path:
            self._learner.learn(learning_item_id)
            exercise_history.append(self._exerciser.test(learning_item_id, self._learner))
        b = sum([v for _, v in self._exerciser.exam(self._learner, *self._learner.target)])
        return exercise_history, b - a, b == len(self._learner.target), None

    def reset(self):
        self._learner = None

    def render(self, mode='human'):
        if mode == "log":
            return "target: %s, state: %s" % (
                self._learner.target, dict(self._exerciser.exam(self._learner, *self._learner.target)))
