# coding: utf-8
# 2021/2/7 @ tongshiwei

from EduSim.utils import dina
from EduSim.Envs.meta import TraitScorer


class TMSScorer(TraitScorer):
    def response_function(self, user_trait, item_trait, *args, **kwargs):
        return dina(user_trait, item_trait["guessing"], item_trait["slipping"])
