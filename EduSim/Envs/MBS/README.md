# Memory-Based Simulators

## Systems
- [x] Exponential Forgetting Curve (EFC)
- [x] Half-Life Regression (HLR)
- [ ] Generalized Power-Law (GPL)

## Env Protocol

- What are provided as environment parameters
    - action space
    - knowledge structure
    - learning item base

- What is recommended, knowledge or item?
    - [ ] knowledge
    - [x] item

### Learner
 
- Which type is the learner, infinity or finite?
    - [x] infinity
    - [ ] finite
- Which mode is the response of the learner, real or trait?
    - [ ] real
    - [x] trait 

### Item
- What are the types of items in this environments?
    - [x] learning item
    - [x] testing item

- What are included in an item?
    - [ ] content
    - [ ] knowledge (or skill)
        - [ ] single
        - [ ] multiple
    - [x] attribute
        - difficulty
        - additional difficulty (in )

### Reward

- Step reward: 0
- Episode reward: $G=\frac{S(T) - S(0)}{S*(T) - S(0)}$

_Eq.(1) in [1]_

## Agent Protocol
- What is recommended, knowledge or item?
    - [ ] knowledge
    - [x] item

## Source Code

The original code can be found in [here](https://github.com/rddy/deeptutor)

### Annotation

We strictly keep the names of variables same with that in the paper, thus some variables have different names against the original code.
For better illustration, we list some important ones as follows (the left is the original code's while the right is ours):

- GPLEnv -> DashEnv 

## Reference

[1] Reddy S, Levine S, Dragan A. Accelerating human learning with deep reinforcement learning[C]//NIPS workshop: teaching machines, robots, and humans. 2017.