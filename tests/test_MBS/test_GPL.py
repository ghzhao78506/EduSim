# coding: utf-8
# 2021/03/12 @ guanhao,zhao

import pytest
from EduSim import GPLEnv, HLREnv, EFCEnv

REWARD = ["likelihood", "log_likelihood"]


@pytest.mark.parametrize("reward", REWARD)
def test_api(reward):
    env = GPLEnv(n_items=30, reward_func=reward, threshold=0.5, seed=None, n_steps=200)
    env.begin_episode()
    env.render(mode="log")
    assert set(env.parameters.keys()) == {'action_space'}


@pytest.mark.parametrize("reward", REWARD)
@pytest.mark.parametrize("n_step", [True, False])
def test_env(reward, n_step):
    from EduSim.Envs.MBS import GPLEnv, MBSAgent, mbs_train_eval
    env = GPLEnv(n_items=30, reward_func=reward, threshold=0.5, seed=None, n_steps=200)
    learner = next(env.learners)
    learner.state
    agent = MBSAgent(env.action_space)
    mbs_train_eval(
        agent,
        env,
        max_steps=200,
        max_episode_num=1,
        level="summary",
        n_step=n_step
    )
