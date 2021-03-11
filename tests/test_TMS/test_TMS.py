# coding: utf-8
# 2019/11/27 @ tongshiwei

import random
from EduSim import TMSEnv
from longling import path_append
import pytest
import gym

NAME = ["binary", "tree"]
MODE = ["no_measurement_error", "with_measurement_error"]


@pytest.mark.parametrize("name", NAME)
@pytest.mark.parametrize("mode", MODE)
def test_api(name, mode):
    env = TMSEnv(name, mode=mode)

    assert set(env.parameters.keys()) == {"knowledge_structure", "action_space"}


@pytest.mark.parametrize("name", NAME)
@pytest.mark.parametrize("mode", MODE)
def test_env(name, mode, tmp_path):
    from EduSim.Envs.TMS import tms_train_eval, TMSAgent

    env = gym.make("TMS-v1", name=name, mode=mode)
    agent = TMSAgent(env.action_space)

    tms_train_eval(
        agent,
        env,
        max_steps=2,
        max_episode_num=10,
        level="summary",
    )
