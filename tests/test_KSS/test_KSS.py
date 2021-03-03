# coding: utf-8
# 2019/11/27 @ tongshiwei

import random
import pytest
from EduSim.Envs.KSS import KSSEnv
from longling import path_append


def test_api(env):
    assert set(env.parameters.keys()) == {"knowledge_structure", "action_space"}


@pytest.mark.parametrize("n_step", [True, False])
def test_env(env, tmp_path, n_step):
    from EduSim.Envs.KSS import kss_train_eval, KSSAgent
    agent = KSSAgent(env.action_space)

    kss_train_eval(
        agent,
        env,
        max_steps=20,
        max_episode_num=10,
        level="summary",
    )
