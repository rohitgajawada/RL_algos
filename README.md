Rohit Gajawada
201401067

Prob1)

a) Cartpole
1) no baseline: python3 policy_gradient.py --baseline='none' --episodes=1000
2) fixed baseline: python3 policy_gradient.py --baseline='fixed' --episodes=1000
3) value state baseline: python3 policy_gradient.py --baseline='valuestate' --episodes=1000

b) Inverted Pendulum
1) no baseline: python3 policy_gradient_cont.py --baseline='none' --episodes=10000
2) fixed baseline: python3 policy_gradient_cont.py --baseline='fixed' --episodes=10000
3) value state baseline: python3 policy_gradient_cont.py --baseline='valuestate' --episodes=10000


Prob2)
Pong:

LFA)
a) Q-Learning with LFA: python3 lin_dqn.py
b) Double Q-Learning with LFA: python3 lin_doubledqn.py
c) PG with LFA: python3 atari_pong_policy_gradient.py --network='lfa'

Neural Network)
a) Q-Learning with neural network: python3 dqn.py --episodes=1000 --algo='dqn'
b) Double Q-Learning with neural network: python3 dqn.py --episodes=1000 --algo='doubledqn'
c) PG with neural network: python3 atari_pong_policy_gradient.py --network='net'
# RL_algos
