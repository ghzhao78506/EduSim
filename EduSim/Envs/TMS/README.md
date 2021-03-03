# Transition Matrix based Simulators

## Env Protocol

- What are provided as environment parameters
    - action space
    - knowledge structure

- What is the action space, knowledge or item?
    - [x] knowledge
    - [ ] item

### Additional

There are two different mode for 

- `no_measurement_error`
- `with_measurement_error`

### Learner
 
- Which type is the learner, infinity or finite?
    - [x] infinity
    - [ ] finite
- Which mode is the response of the learner, real or hidden?
    - [ ] real
    - [x] hidden 

### Item
- What are the types of items in this environments?
    - [x] learning item
    - [x] testing item

- What are included in an item?
    - [ ] content
    - [x] knowledge (or skill)
        - [x] single
        - [ ] multiple
    - [x] attribute
        - guessing
        - slipping

### Reward

- Step reward: $R(t) = \sum_{k=1}^{K}[\alpha_k(t+1)-\alpha_k(t)]$
- Episode reward: $G=\sum_{t=0}^{T-1}R(t)$

_Find these two equations in [1] (Eq.(1) and Eq.(2))_


## Agent Protocol
- What is recommended (i.e., what is the action space), knowledge or item?
    - [x] knowledge
    - [ ] item


## Systems

* binary: Study I in Tang et al. [1]
* tree: Study II in Tang et al. [1]


## Reference

[1] Tang X, Chen Y, Li X, et al. A reinforcement learning approach to personalized learning recommendation systems[J]. British Journal of Mathematical and Statistical Psychology, 2019, 72(1): 108-135.

