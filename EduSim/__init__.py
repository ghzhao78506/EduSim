# coding: utf-8
# create by tongshiwei on 2019/6/25

from gym.envs.registration import register
from .Envs import *
from .SimOS import train_eval, MetaAgent
from .spaces import *
from .ItemBase import ItemBase

# register(
#     id='KSS-v1',
#     entry_point='EduSim.Envs:KSSEnv',
# )

register(
    id='TMS-v1',
    entry_point='EduSim.Envs:TMSEnv',
)
