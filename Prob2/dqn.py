import matplotlib
matplotlib.use('Agg')
import gym
import numpy as np
import math
import torch
import torch.optim as optim
import sys
import argparse
from wrappers_2 import make_atari, wrap_deepmind, wrap_pytorch
from utils import replaybuffer, plot, QNet


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env', default="PongNoFrameskip-v4", type=str, help='gym environment')
parser.add_argument('--algo', default="doubledqn", type=str, help='algorithm')
parser.add_argument('--type', default='discrete', type=str, help='action type')
parser.add_argument('--seed', default=123, type=int, help='random seed')

parser.add_argument('--batchsize', default=32, type=int, help='random seed')
parser.add_argument('--numframes', default=2000000, type=int, help='num frames')
parser.add_argument('--episodes', default=2000, type=int, help='number of episodes')
parser.add_argument('--horizon', default=2000, type=int, help='number of time steps')
parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
parser.add_argument('--epsilon_start', default=1.0, type=float, help='eps start')
parser.add_argument('--epsilon_final', default=0.01, type=float, help='eps end')
parser.add_argument('--epsilon_decay', default=30000, type=int, help='eps decay')

parser.add_argument('--plotfreq', default=10000, type=int)
parser.add_argument('--targetupdfreq', default=1000, type=int)
parser.add_argument('--updthresh', default=10000, type=int)


args = parser.parse_args()
torch.manual_seed(args.seed)
print(args)
algo = args.algo

env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)

model = QNet(env.observation_space.shape, env.action_space.n)
model = model.cuda()
if algo == 'doubledqn':
	target_model  = QNet(env.observation_space.shape, env.action_space.n)
	target_model = target_model.cuda()

print(model)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

replay_buffer = replaybuffer(100000)

if args.algo == 'doubledqn':
	target_model.load_state_dict(model.state_dict())

losses = []
all_rewards = []
episode_reward = 0
last = 0
running_rew = 0
run_rewards = []
ct = 0

state = env.reset()
for idx in range(1, args.numframes + 1):
	epsilon = max(1 - 0.9 * (ct / 30.0), 0.1)
	action = model.act(state, epsilon)

	next_state, reward, done, _ = env.step(action)
	replay_buffer.push(state, action, reward, next_state, done)

	state = next_state
	episode_reward += reward
	sys.stdout.flush()

	if done:
		state = env.reset()
		all_rewards.append(episode_reward)

		ct += 1
		running_rew += episode_reward
		runval = running_rew / (1.0 * ct)
		run_rewards.append(runval)

		print(episode_reward, runval, idx - last, epsilon)
		last = idx
		episode_reward = 0

	if idx > args.updthresh:
		if algo == 'dqn':
			s, act, r, n_s, d = replay_buffer.sample(args.batchsize)

			s = torch.FloatTensor(np.float32(s)).cuda()
			n_s = torch.FloatTensor(np.float32(n_s)).cuda()
			act = torch.LongTensor(act).cuda()
			r = torch.FloatTensor(r).cuda()
			d = torch.FloatTensor(d).cuda()

			q_vals = model(s)
			next_q_vals = model(n_s)

			q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)
			next_q_val = next_q_vals.max(1)[0]
			exp_q_val = r + args.gamma * next_q_val * (1 - d)

			loss = (q_val - exp_q_val.detach().data.cuda()).pow(2).mean()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		elif algo == 'doubledqn':
			s, act, r, n_s, d = replay_buffer.sample(args.batchsize)

			s = torch.FloatTensor(np.float32(s)).cuda()
			n_s = torch.FloatTensor(np.float32(n_s)).cuda()
			act = torch.LongTensor(act).cuda()
			r = torch.FloatTensor(r).cuda()
			d = torch.FloatTensor(d).cuda()

			q_vals = model(s)
			next_q_vals = model(n_s)
			next_q_state_vals = target_model(n_s)

			q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)
			next_q_val = next_q_state_vals.gather(1, torch.max(next_q_vals, 1)[1].unsqueeze(1)).squeeze(1)
			exp_q_val = r + args.gamma * next_q_val * (1 - d)

			loss = (q_val - exp_q_val.detach().data.cuda()).pow(2).mean()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		losses.append(loss.item())

	if idx % args.plotfreq == 0:
		plot(idx, all_rewards, run_rewards, algo)

	if idx % args.targetupdfreq == 0 and algo == 'doubledqn':
		target_model.load_state_dict(model.state_dict())
