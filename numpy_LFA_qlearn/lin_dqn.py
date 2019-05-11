import gym
import argparse
from util import replaybuffer
import random
import numpy as np
import matplotlib.pyplot as plt

class DQN():
    def __init__(self, insize, action_space):
        super(DQN, self).__init__()
        self.action_space = action_space
        self.weights = np.array([np.zeros(insize) for i in range(action_space)])

    def forward(self, x):
        q_vals = np.matmul(self.weights, x)
        return q_vals

    def act(self, state, epsilon):
        if random.random() > epsilon - epsilon/self.action_space:
            q_val = self.forward(state)
            action = np.argmax(q_val)
        else:
            action = np.random.randint(self.action_space)
        return action

def prepro(x):

    x = x[35: 195]
    x = x[::2, ::2, 0]
    x[x == 144] = 0
    x[x == 109] = 0
    x[x != 0] = 1
    return x.astype(np.float).ravel()

def plot(running_rews, type):
	plt.figure(figsize=(5,5))
	plt.title('running reward')
	plt.plot(running_rews)
	plt.show()


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env', default='Pong-v0', type=str, help='gym environment')
parser.add_argument('--algo', default='dqn', type=str, help='algo: dqn/doubledqn')
parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
parser.add_argument('--seed', default=123, type=int, help='random seed')
parser.add_argument('--episodes', default=10000, type=int, help='number of episodes')
parser.add_argument('--horizon', default=50000, type=int, help='number of time steps')
parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')

parser.add_argument('--lr_start', default=0.01, type=float)
parser.add_argument('--lr_end', default=0.0005, type=float)
parser.add_argument('--eps_start', default=1, type=float)
parser.add_argument('--eps_end', default=0.1, type=float)
parser.add_argument('--nsteps', default=100000, type=int, help='total steps')

parser.add_argument('--framehistory', default=1, type=int, help='number of images into network')
parser.add_argument('--buffersize', default=1000, type=int, help='replay buffer size')
parser.add_argument('--batchsize', default=4, type=int, help='replay buffer size')


args = parser.parse_args()
print(args)

env = gym.make(args.env)
# env = MaxAndSkipEnv(env, skip=4)
# env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), overwrite_render=True)

lr = args.lr_start

dqn = DQN(6400, env.action_space.n)
eps_vals = np.linspace(args.eps_end, args.eps_start, args.nsteps)

total_rewards = []
count = 0
run_avg = 0
running_avg_rew = []

for episode in range(args.episodes):

    state = env.reset()
    # print(state)
    state = prepro(state)
    # print(state)
    # state = np.reshape(state, (6400)) / 255.0

    eps_reward = 0
    for t in range(args.horizon):

        if count < args.nsteps:
            epsilon = eps_vals[-(count+1)]
        else:
            epsilon = 0.15

        action = dqn.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        next_state = prepro(next_state)
        # print(np.sum(next_state))
        # env.render()
        # next_state = np.reshape(next_state, (6400)) / 255.0

        old_state = state
        state = next_state

        # next_action = dqn.actmax(state)
        q_vals = dqn.forward(old_state)
        next_q_vals = dqn.forward(state)
        max_next_q_val = np.max(next_q_vals)

        #Update equation
        # print(lr, reward, args.gamma, max_next_q_val, q_vals[action], action)
        # print(dqn.weights)
        # print(q_vals[action])
        dqn.weights[action] += lr * (reward + args.gamma * max_next_q_val - q_vals[action]) * old_state
        # normfact = np.sum(dqn.weights[:, action])
        eps_reward += reward
        count += 1

        if done:
            total_rewards.append(eps_reward)
            eps_reward = 0
            break

        if count < args.batchsize:
            continue

    run_avg += total_rewards[-1]
    runval = run_avg / (episode + 1.0)
    running_avg_rew.append(runval)

    print("Epi", episode, "Epsilon", epsilon, "Rewards", total_rewards[-1], "Time", t)

plot(running_avg_rew, args.algo)
