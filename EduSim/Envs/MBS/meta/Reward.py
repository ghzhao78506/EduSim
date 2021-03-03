# coding: utf-8
# 2020/5/13 @ tongshiwei
import numpy as np


class Reward(object):
    def __init__(self, reward_func="likelihood"):
        self.reward_func = reward_func

    @staticmethod
    def likelihood(probabilities):
        return np.asarray(probabilities).mean()

    @staticmethod
    def log_likelihood(probabilities, eps=1e-9):
        return np.log(eps + np.asarray(probabilities)).mean()

    def __call__(self, probabilities, *args, **kwargs):
        if self.reward_func == "likelihood":
            return self.likelihood(probabilities)
        elif self.reward_func == "log_likelihood":
            return self.log_likelihood(probabilities)
        else:
            raise TypeError("unknown reward function: %s" % self.reward_func)
