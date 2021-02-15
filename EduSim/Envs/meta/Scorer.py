# coding: utf-8
# 2021/1/28 @ tongshiwei


class Scorer(object):
    def __call__(self, user, item, *args, **kwargs) -> ...:
        raise NotImplemented


class RealScorer(Scorer):
    def answer_scoring(self, user_response, item_truth, *args, **kwargs):
        raise NotImplemented

    def __call__(self, user_response, item_truth, *args, **kwargs):
        return self.answer_scoring(user_response, item_truth, *args, **kwargs)


class RealChoiceScorer(RealScorer):
    def answer_scoring(self, user_response, item_truth, *args, **kwargs):
        return user_response == item_truth


class HiddenScorer(Scorer):
    def __init__(self):
        super(HiddenScorer, self).__init__()

    def response_function(self, user_trait, item_trait, *args, **kwargs):
        raise NotImplemented

    def __call__(self, user_trait, item_trait, *args, **kwargs):
        return self.response_function(user_trait, item_trait, *args, **kwargs)