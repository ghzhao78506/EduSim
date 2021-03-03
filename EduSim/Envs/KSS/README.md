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

- Step reward: 0
- Episode reward: $G=\frac{S(T) - S(0)}{S*(T) - S(0)}$

_Eq.(1) in [1]_

## Agent Protocol
- What is recommended, knowledge or item?
    - [ ] knowledge
    - [x] item


## Reference

[1] Liu Q, Tong S, Liu C, et al. Exploiting cognitive structure for adaptive learning[C]//Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019: 627-635.

