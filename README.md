# EduSim
[![PyPI](https://img.shields.io/pypi/v/EduSim)](https://pypi.python.org/pypi/EduSim)
[![Build Status](https://www.travis-ci.org/tswsxk/EduSim.svg?branch=master)](https://www.travis-ci.org/tswsxk/EduSim)
[![codecov](https://codecov.io/gh/tswsxk/EduSim/branch/master/graph/badge.svg)](https://codecov.io/gh/tswsxk/EduSim)

EduSim is a platform for constructing simulation environments for education recommender systems (ERSs) 
that naturally supports sequential interaction with learners. 
Meanwhile, EduSim allows the creation of new environments that reflect particular aspects of learning elements, 
such as learning behavior of learners, knowledge structure of concepts and so on.

If you are using this package for your research, please cite our paper [1].

Refer to our [website](http://base.ustc.edu.cn/) and [github](https://github.com/bigdata-ustc) for our publications and more projects

## Installation
```bash
pip install EduSim
```

## Quick Start
See the examples in examples directory

## List of Environment

There are three kinds of Environments, which differs in learner capacity growth model:
* Pattern Based Simulators (PBS): the capacity growth model is designed by human experts;
* Data Driven Simulators (DDS): the capacity growth model is learned from real data;
* Hybrid Simulators (HS): the capacity growth model is learned from real data with some expert rule limitation;

We currently provide the following environments:

Name | Kind | Notation
-|-|-
[TMS-v1](docs/Env.md) | PBS | Transition Matrix based Simulator (TMS), which is used in [1,2,3]
[MBS-v0](docs/Env.md) | PBS | Memory Based Simulator (MBS), which is used in [4]
[KSS-v2](docs/Env.md) | PBS | Knowledge Structure based Simulator (KSS), which is used in [5]

To construct your own environment, refer to [Env.md](docs/Env.md)

Declaration: if you are using ``TMS`` and ``MBS``, referring to the citations is suggested.


## utils

### Visualization

By default, we use ``tensorboard`` to help visualize the reward of each episode, see demos in ``scripts`` and use
```sh
tensorboard --logdir /path/to/logs
```
to see the visualization result.

## Reference
[1] Tang X, Chen Y, Li X, et al. A reinforcement learning approach to personalized learning recommendation systems[J]. British Journal of Mathematical and Statistical Psychology, 2019, 72(1): 108-135.

[2] 

[3] 

[4] Reddy S, Levine S, Dragan A. Accelerating human learning with deep reinforcement learning[C]//NIPS workshop: teaching machines, robots, and humans. 2017.

[5] Qi Liu, Shiwei Tong, Chuanren Liu, Hongke Zhao, Enhong Chen, HaipingMa,&ShijinWang.2019.Exploiting Cognitive Structure for Adaptive Learning.InThe 25th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (KDD’19)
```bibtex
@inproceedings{DBLP:conf/kdd/LiuTLZCMW19,
  author    = {Qi Liu and
               Shiwei Tong and
               Chuanren Liu and
               Hongke Zhao and
               Enhong Chen and
               Haiping Ma and
               Shijin Wang},
  title     = {Exploiting Cognitive Structure for Adaptive Learning},
  booktitle = {Proceedings of the 25th {ACM} {SIGKDD} International Conference on
               Knowledge Discovery {\&} Data Mining, {KDD} 2019, Anchorage, AK,
               USA, August 4-8, 2019},
  pages     = {627--635},
  year      = {2019},
  crossref  = {DBLP:conf/kdd/2019},
  url       = {https://doi.org/10.1145/3292500.3330922},
  doi       = {10.1145/3292500.3330922},
  timestamp = {Mon, 26 Aug 2019 12:44:14 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/kdd/LiuTLZCMW19},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
