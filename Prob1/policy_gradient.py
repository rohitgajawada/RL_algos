import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import argparse
import networks

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env', default='CartPole-v0', type=str, help='gym environment')
parser.add_argument('--type', default='discrete', type=str, help='action type')
parser.add_argument('--baseline', default='none', type=str, help='none | fixed | valuestate')
parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
parser.add_argument('--seed', default=123, type=int, help='random seed')
parser.add_argument('--episodes', default=2000, type=int, help='number of episodes')
parser.add_argument('--horizon', default=2000, type=int, help='number of time steps')
parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
parser.add_argument('--printfreq', default=50, type=float, help='rewards discount factor')

args = parser.parse_args()
torch.manual_seed(args.seed)
print(args)

def plot(frame_idx, rewards, advantages, running_rews, type):
	plt.figure(figsize=(20,5))
	plt.subplot(131)
	plt.title('rewards ' + type)
	plt.plot(rewards)
	plt.subplot(132)
	plt.title('advantage')
	plt.plot(advantages)
	plt.subplot(133)
	plt.title('running reward')
	plt.plot(running_rews)
	plt.show()

env = gym.make(args.env)
if args.baseline != 'valuestate':
	policy_net = networks.PolicyNet(env.observation_space.shape[0], env.action_space.n)
	optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
	print(policy_net)
else:
	policy_net = networks.CombinedNet(env.observation_space.shape[0], env.action_space.n)
	optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
	print(policy_net)

policy_net.policy_history = torch.Tensor()
policy_net.value_history = torch.Tensor()
policy_net.reward_episode = []
policy_net.reward_history = []
losses = []
all_rewards = []
advantages = []
running_avg_rew = []
run_avg = 0

for episode in range(args.episodes):
	state = env.reset()
	done = False

	for time in range(args.horizon):

		#Action sampling
		state = torch.from_numpy(state).type(torch.FloatTensor)

		if args.baseline != 'valuestate':
			state_out = policy_net(state)
		else:
			state_out, value_out = policy_net(state)
			value_out = value_out.view(-1)

		phi_s = Categorical(state_out)
		action = phi_s.sample()

		if policy_net.policy_history.dim() != 0 and time > 0:
			log_prob = phi_s.log_prob(action)
			policy_net.policy_history = torch.cat([policy_net.policy_history, log_prob])

			if args.baseline == 'valuestate':
				policy_net.value_history = torch.cat([policy_net.value_history, value_out])

		else:
			log_prob = phi_s.log_prob(action)
			policy_net.policy_history = log_prob

			if args.baseline == 'valuestate':
				policy_net.value_history = value_out

		#Get next s,r pair
		state, reward, done, _ = env.step(action.item())
		# env.render(mode='rgb_array')

		policy_net.reward_episode.append(reward)
		if done:
			break

	#Update the policy
	total_r = 0
	rewards = []
	for r in policy_net.reward_episode[::-1]:
		total_r = r + args.gamma * total_r
		rewards.insert(0, total_r)

	rewards = torch.FloatTensor(rewards)

	if args.baseline == "fixed":
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
		# rewards = (rewards - rewards.mean())
	elif args.baseline == "valuestate":
		rewards = (rewards - rewards.mean())
		policy_net.value_history = (policy_net.value_history - policy_net.value_history.mean())

		# print(rewards, policy_net.value_history)

		# rewards = (rewards - policy_net.value_history)

	adv_val = torch.sum(rewards)
	policy_losses = []
	value_losses = []


	if args.baseline == 'valuestate':
		for i in range(rewards.size(0)):
			log_prob = policy_net.policy_history[i]
			value = policy_net.value_history[i]
			r = rewards[i]

			reward = r - value.item()
			policy_losses.append(-log_prob * reward)
			value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))

		optimizer.zero_grad()
		loss = torch.stack(policy_losses).sum()
		loss += torch.stack(value_losses).sum()
		loss.backward()
		optimizer.step()

	else:
		policy_loss = -1 * torch.sum(torch.mul(rewards, policy_net.policy_history), -1)
		loss = policy_loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	reward_sum = np.sum(policy_net.reward_episode)
	policy_net.reward_history.append(reward_sum)
	policy_net.policy_history = torch.Tensor()
	policy_net.reward_episode= []

	run_avg += reward_sum

	all_rewards.append(reward_sum)
	losses.append(loss.data[0])
	advantages.append(adv_val.data[0])
	runval = run_avg / (episode + 1.0)
	running_avg_rew.append(runval)

	print(episode, "Reward:", reward_sum, "Adv", adv_val, "Run", runval)

plot(episode + 1, all_rewards, advantages, running_avg_rew, args.baseline)
