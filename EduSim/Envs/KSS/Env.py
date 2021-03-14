# coding: utf-8
# 2020/4/30 @ tongshiwei
from copy import deepcopy
import networkx as nx
import random
from EduSim.Envs.meta import Env

import numpy as np
import pytest
from EduSim.Envs.KSS.meta.Learner import LearnerGroup, Learner
from EduSim.Envs.shared.KSS_KES import episode_reward
from EduSim.spaces import ListSpace
from .meta import KSSItemBase, KSSScorer
from .utils import load_environment_parameters

__all__ = ["KSSEnv"]


class KSSEnv(Env):
    def __init__(self, seed=None, initial_step=20):
        self.random_state = np.random.RandomState(seed)

        parameters = load_environment_parameters()
        self.knowledge_structure = parameters["knowledge_structure"]
        self._item_base = KSSItemBase(
            parameters["knowledge_structure"],
            parameters["learning_order"],
            items=parameters["items"]
        )
        self.learning_item_base = deepcopy(self._item_base)
        self.learning_item_base.drop_attribute()
        self.test_item_base = self._item_base
        self.scorer = KSSScorer(parameters["configuration"].get("binary_scorer", True))

        self.action_space = ListSpace(self.learning_item_base.item_id_list, seed=seed)

        self.learners = LearnerGroup(self.knowledge_structure, seed=seed)

        self._order_ratio = parameters["configuration"]["order_ratio"]
        self._review_times = parameters["configuration"]["review_times"]
        self._learning_order = parameters["learning_order"]

        self._topo_order = list(nx.topological_sort(self.knowledge_structure))
        self._initial_step = parameters["configuration"]["initial_step"] if initial_step is None else initial_step

        self._learner = None
        self._initial_score = None
        self._exam_reduce = "sum" if parameters["configuration"].get("exam_sum", True) else "ave"

    @property
    def parameters(self) -> dict:
        return {
            "knowledge_structure": self.knowledge_structure,
            "action_space": self.action_space,
            "learning_item_base": self.learning_item_base
        }

    def _initial_logs(self, learner: Learner):
        logs = []

        if random.random() < self._order_ratio:
            while len(logs) < self._initial_step:
                if logs and logs[-1][1] == 1 and len(
                        set([e[0] for e in logs[-3:]])) > 1:
                    for _ in range(self._review_times):
                        if len(logs) < self._initial_step - self._review_times:
                            learning_item_id = logs[-1][0]
                            test_item_id, score = self.learn_and_test(learner, learning_item_id)
                            logs.append([test_item_id, score])
                        else:
                            break
                    learning_item_id = logs[-1][0]
                elif logs and logs[-1][1] == 0 and random.random() < 0.7:
                    learning_item_id = logs[-1][0]
                elif random.random() < 0.9:
                    for knowledge in self._topo_order:
                        test_item_id = self.test_item_base.knowledge2item[knowledge].id
                        if learner.response(self.test_item_base[test_item_id]) < 0.6:
                            break
                    else:  # pragma: no cover
                        break
                    learning_item_id = test_item_id
                else:
                    learning_item_id = self.random_state.choice(list(self.learning_item_base.index))

                test_item_id, score = self.learn_and_test(learner, learning_item_id)
                logs.append([test_item_id, score])
        else:
            while len(logs) < self._initial_step:
                if random.random() < 0.9:
                    for knowledge in self._learning_order:
                        test_item_id = self.test_item_base.knowledge2item[knowledge].id
                        if learner.response(self.test_item_base[test_item_id]) < 0.6:
                            break
                    else:
                        break
                    learning_item_id = test_item_id
                else:
                    learning_item_id = self.random_state.choice(self.learning_item_base.index)

                item_id, score = self.learn_and_test(learner, learning_item_id)
                logs.append([item_id, score])

        learner.update_logs(logs)

    def learn_and_test(self, learner: Learner, item_id):
        learning_item = self.learning_item_base[item_id]
        learner.learn(learning_item)
        test_item_id = item_id
        test_item = self.test_item_base[test_item_id]
        score = self.scorer(learner.response(test_item), test_item.attribute)
        return item_id, score

    def _exam(self, learner: Learner, detailed=False, reduce=None) -> (dict, int, float):
        if reduce is None:
            reduce = self._exam_reduce
        knowledge_response = {}
        for test_knowledge in learner.target:
            item = self.test_item_base.knowledge2item[test_knowledge]
            knowledge_response[test_knowledge] = [item.id, self.scorer(learner.response(item), item.attribute)]
        if detailed:
            return knowledge_response
        elif reduce == "sum":
            return np.sum([v for _, v in knowledge_response.values()])
        elif reduce in {"mean", "ave"}:
            return np.average([v for _, v in knowledge_response.values()])
        else:
            raise TypeError("unknown reduce type %s" % reduce)

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)
        self._initial_logs(self._learner)
        self._initial_score = self._exam(self._learner)
        return self._learner.profile, self._exam(self._learner, detailed=True)

    def end_episode(self, *args, **kwargs):
        observation = self._exam(self._learner, detailed=True)
        initial_score, self._initial_score = self._initial_score, None
        final_score = self._exam(self._learner)
        reward = episode_reward(initial_score, final_score, len(self._learner.target))
        done = final_score == len(self._learner.target)
        info = {"initial_score": initial_score, "final_score": final_score}
        self._learner = None

        return observation, reward, done, info

    def step(self, learning_item_id, *args, **kwargs):
        a = self._exam(self._learner)
        observation = self.learn_and_test(self._learner, learning_item_id)
        b = self._exam(self._learner)
        return observation, b - a, b == len(self._learner.target), None

    def n_step(self, learning_path, *args, **kwargs):
        exercise_history = []
        a = self._exam(self._learner)
        for learning_item_id in learning_path:
            item_id, score = self.learn_and_test(self._learner, learning_item_id)
            exercise_history.append([item_id, score])
        b = self._exam(self._learner)
        return exercise_history, b - a, b == len(self._learner.target), None

    def reset(self):
        self._learner = None

    def render(self, mode='human'):
        if mode == "log":
            return "target: %s, state: %s" % (
                self._learner.target, dict(self._exam(self._learner))
            )
