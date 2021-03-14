# coding: utf-8
# 2020/4/29 @ tongshiwei

from pprint import pformat

from longling import path_append, abs_current_dir
from EduSim.Envs.meta import Env
from copy import deepcopy

from .meta import LearnerGroup, TMSTestItemBase, TMSLearningItemBase, TMSScorer
from .utils import load_environment_parameters
from EduSim.spaces import ListSpace

__all__ = ["TMSEnv"]

ROOT = path_append(abs_current_dir(__file__), "meta_data")

ENV_META = {
    "binary": path_append(ROOT, "binary", to_str=True),
    "tree": path_append(ROOT, "tree", to_str=True),
}

NO_MEASUREMENT_ERROR = 0
MEASUREMENT_ERROR = 1

NAME = [
    "binary",
    "tree",
]

MODE = {
    "no_measurement_error": NO_MEASUREMENT_ERROR,
    "with_measurement_error": MEASUREMENT_ERROR,
}


class TMSEnv(Env):
    """
    Example
    -------
    >>> tms = TMSEnv(name="binary",parameters={"a":1})
    >>> [i for i in tms.parameters.keys()]
    ['knowledge_structure', 'action_space']
    """
    def __init__(self, name, mode="with_measurement_error", seed=None, parameters: dict = None):
        """

        Parameters
        ----------
        name:
            binary or tree
        parameters: dict
            * test_item_for_each_skill
        mode
        """
        super(TMSEnv, self).__init__()

        directory = ENV_META.get(name, name)
        environment_parameters = load_environment_parameters(directory)
        if parameters is not None:
            environment_parameters.update(parameters)

        self.knowledge_structure = environment_parameters["knowledge_structure"]
        self.learners = LearnerGroup(
            transition_matrix=environment_parameters["transition_matrix"],
            state2vector=environment_parameters["state2vector"],
            initial_states=environment_parameters["initial_states"],
            seed=seed
        )
        self._skill_num = environment_parameters["configuration"]["skill_num"]
        self._test_item_for_each_skill = environment_parameters["configuration"]["test_item_for_each_skill"]
        self.mode_str = mode
        self.mode = MODE[mode]
        self.learning_item_base = TMSLearningItemBase({
            "d%s" % i: {"knowledge": i} for i in range(self._skill_num)
        })
        self.test_item_base = TMSTestItemBase(
            self._skill_num,
            self._test_item_for_each_skill,
            seed=seed
        )
        self.scorer = TMSScorer()
        self._learner = None
        self._begin_state = None

        self.action_space = ListSpace(self.learning_item_base.item_id_list, seed=seed)

    def __repr__(self):
        return pformat({
            "skill_num": self._skill_num,
            "test_item_for_each_skill": self._test_item_for_each_skill,
            "mode": self.mode_str
        })

    @property
    def parameters(self) -> dict:
        return {
            "knowledge_structure": self.knowledge_structure,
            "action_space": self.action_space,
        }

    def render(self, mode='human'):
        if mode == "log":
            return "state: %s" % str(self._learner.state)

    def reset(self):
        self._begin_state = None

    def gen_obs(self):
        # section 4.1.1
        if self.mode == NO_MEASUREMENT_ERROR:
            observation = self._learner.state
        else:
            observation = [
                self.scorer(self._learner.state[item.knowledge], item.attribute)
                for item in self.test_item_base
            ]
        return observation

    def step(self, learning_item_id, *args, **kwargs):
        proficiency = sum(self._learner.state)
        self._learner.learn(self.learning_item_base[learning_item_id])

        # Equation (1)
        reward = sum(self._learner.state) - proficiency

        observation = self.gen_obs()

        return observation, reward, len(self._learner.state) == sum(
            self._learner.state), None

    def n_step(self, learning_path, *args, **kwargs):
        return zip(*[self.step(learning_item) for learning_item in learning_path])

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)
        self._begin_state = deepcopy(self._learner.state)
        return self._learner.profile, self.gen_obs()

    def end_episode(self, *args, **kwargs):
        observation = self.gen_obs()

        reward = sum(self._learner.state) - sum(self._begin_state)
        return observation, reward, len(self._learner.state) == sum(self._learner.state), None
