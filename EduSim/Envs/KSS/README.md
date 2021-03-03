# Knowledge Structure based Simulators

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
    - [x] test item

- Is the learning item base same with the test item base?
    - [ ] Yes
    - [x] No

- The difference between learning item base and test item base?
    - [ ] completely
    - [x] property
        - [ ] knowledge
        - [ ] content
        - [x] attribute: learning item do not have difficulty

- What are included in an item?
    - [ ] content
    - [x] knowledge
        - [x] single
        - [ ] multiple
    - [x] attribute
        - difficulty

- What is the relation between item and knowledge in learning item?
    - [x] one-to-one
    - [ ] one-to-many
    - [ ] many-to-one
    - [ ] many-to-many
    
- What is the relation between item and knowledge in test item?
    - [x] one-to-one
    - [ ] one-to-many
    - [ ] many-to-one
    - [ ] many-to-many

### Reward

- Step reward: 0
- Episode reward: $G=\frac{S(T) - S(0)}{S*(T) - S(0)}$

_Eq.(1) in [1]_

## Agent Protocol
- What is recommended, knowledge or item?
    - [ ] knowledge
    - [x] item

## Original Code

This is the original implementation in [1]


## Reference

[1] Liu Q, Tong S, Liu C, et al. Exploiting cognitive structure for adaptive learning[C]//Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019: 627-635.

