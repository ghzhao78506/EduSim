# Transition Matrix based Simulators

## Env Protocol

- What are provided as environment parameters
    - action space
    - knowledge structure

- What is the action space, knowledge or item?
    - [ ] knowledge
    - [x] item

### Additional

There are two different mode for 

- `no_measurement_error`
- `with_measurement_error`

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
    - [x] test item

- Is the learning item base same with the test item base?
    - [ ] Yes
    - [x] No 

- What are included in an item?
    - [ ] content
    - [x] knowledge
        - [x] single
        - [ ] multiple
    - [x] attribute
        - guessing
        - slipping
        
- What is the relation between item and knowledge in learning item?
    - [x] one-to-one
    - [ ] one-to-many
    - [ ] many-to-one
    - [ ] many-to-many
    
- What is the relation between item and knowledge in test item?
    - [ ] one-to-one
    - [ ] one-to-many
    - [x] many-to-one
    - [ ] many-to-many

### Reward

- Step reward: $R(t) = \sum_{k=1}^{K}[\alpha_k(t+1)-\alpha_k(t)]$
- Episode reward: $G=\sum_{t=0}^{T-1}R(t)$

_Find these two equations in [1] (Eq.(1) and Eq.(2))_


## Agent Protocol
- What is recommended (i.e., what is the action space), knowledge or item?
    - [ ] knowledge
    - [x] item


## Systems

* binary: Study I in Tang et al. [1]
* tree: Study II in Tang et al. [1]


## Original Code

Tang et al. [1] did not provide source code.

## Reference

[1] Tang X, Chen Y, Li X, et al. A reinforcement learning approach to personalized learning recommendation systems[J]. British Journal of Mathematical and Statistical Psychology, 2019, 72(1): 108-135.

