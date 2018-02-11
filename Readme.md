## pong rl
pong with policy gradient, implemented in tensorflow.

the goal is to find policy $\pi s_i \mapsto a_i$ that maximizes expected rewards. in practice, this means maximizing $log p(a|s, \theta) \times r$.

## usage
run `python main.py`. 
