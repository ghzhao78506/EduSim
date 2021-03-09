import gym
from EduSim.Envs.MBS import GPLEnv, MBSAgent, mbs_train_eval

env: GPLEnv = gym.make("MBS-GPL-v0", seed=10, n_steps=200)
agent = MBSAgent(env.action_space)

from longling import set_logging_info
set_logging_info()
mbs_train_eval(
    agent,
    env,
    max_steps=200,
    max_episode_num=100,
    level="summary",
)
print("done")