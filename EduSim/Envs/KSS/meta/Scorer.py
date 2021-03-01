# coding: utf-8
# 2021/2/7 @ tongshiwei

from EduSim.utils import irt
from EduSim.Envs.meta import HiddenScorer


class KSSScorer(HiddenScorer):
    def response_function(self, user_trait, item_trait, *args, **kwargs):
        return irt(user_trait, item_trait["difficulty"])
