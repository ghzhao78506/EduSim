# coding: utf-8
# 2021/2/7 @ tongshiwei
from tensorboardX import SummaryWriter
import logging
from EduSim.utils import board_episode_callback
from EduSim.SimOS import train_eval
from .Env import TMSEnv


def tms_train_eval(agent, env: TMSEnv, max_steps: int = None, max_episode_num: int = None, n_step=False,
                   train=False,
                   logger=logging, level="episode", board_dir=None):
    """

    Parameters
    ----------
    agent
    env
    max_steps
    max_episode_num
    n_step
    train
    logger
    level
    board_dir: the directory to hold tensorboard result
        use ``tensorboard --logdir $board_dir`` to see the result

    Returns
    -------

    """

    assert max_episode_num is not None, "infinity environment, max_episode_num should be set"

    def summary_callback(rewards, infos, logger):
        expected_reward = sum(rewards) / len(rewards)

        logger.info("Expected Reward: %s" % expected_reward)

        return expected_reward, infos

    sw = None
    if board_dir is not None:
        sw = SummaryWriter(board_dir)

        def episode_callback(episode, reward, *args):
            return board_episode_callback(episode, reward, sw)

    else:
        episode_callback = None

    train_eval(
        agent, env,
        max_steps, max_episode_num, n_step, train,
        logger, level,
        episode_callback=episode_callback,
        summary_callback=summary_callback,
    )

    if board_dir:
        sw.close()
