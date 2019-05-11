import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random

class PolicyNet(nn.Module):
	def __init__(self, state_space, action_space):
		super(PolicyNet, self).__init__()
		self.state_space = state_space
		self.action_space = action_space
		self.fc1 = nn.Linear(state_space, 128)
		self.fc2 = nn.Linear(128, action_space)

	def forward(self, x):
		x = x.view(1, -1)
		x = F.relu(self.fc1(x))
		x = F.softmax(self.fc2(x), dim=-1)
		return x

class CombinedNet(nn.Module):
	def __init__(self, state_space, action_space):
		super(CombinedNet, self).__init__()
		self.state_space = state_space
		self.action_space = action_space
		self.fc1 = nn.Linear(state_space, 128)
		self.q_fc = nn.Linear(128, action_space)

		self.v_fc1 = nn.Linear(128, 128)
		self.v_fc2 = nn.Linear(128, 1)

	def forward(self, x):
		x = x.view(1, -1)
		x = F.relu(self.fc1(x))
		q_vals = F.softmax(self.q_fc(x), dim=-1)

		state_vals = F.relu(self.v_fc1(x))
		state_vals = F.tanh(self.v_fc2(state_vals))

		return q_vals, state_vals

class ContPolicyNet(nn.Module):
	def __init__(self, state_space, action_dim, action_lim):
		super(ContPolicyNet, self).__init__()
		self.state_space = state_space
		self.action_dim = action_dim
		self.action_lim = torch.FloatTensor([action_lim])
		self.fc1 = nn.Linear(state_space, 128)
		self.fc2 = nn.Linear(128, action_dim)
		self.fc2.weight.data.uniform_(-0.003, 0.003)

	def forward(self, x):
		x = x.view(1, -1)
		x = F.relu(self.fc1(x))
		action = F.tanh(self.fc2(x))
		action = action * self.action_lim
		return action

class ContCombinedNet(nn.Module):
	def __init__(self, state_space, action_dim, action_lim):
		super(ContCombinedNet, self).__init__()
		self.state_space = state_space
		self.action_dim = action_dim
		self.action_lim = torch.FloatTensor([action_lim])

		self.fc1 = nn.Linear(state_space, 128)
		self.q_fc = nn.Linear(128, action_dim)
		self.q_fc.weight.data.uniform_(-0.003, 0.003)

		self.v_fc = nn.Linear(128, 1)

	def forward(self, x):
		x = x.view(1, -1)
		x = F.relu(self.fc1(x))
		action = F.tanh(self.q_fc(x))
		action = action * self.action_lim

		state_val = F.tanh(self.v_fc(x))

		return action, state_val

class DQN(nn.Module):
	def __init__(self, in_channels, action_space):
		super(DQN, self).__init__()
		self.relu = nn.ReLU()
		self.action_space = action_space
		self.fc1 = nn.Linear(7056, out_features=action_space) #7056

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		return x

	def act(self, state, epsilon):
		if random.random() > epsilon:
			q_val = self.forward(state)
			action = q_val.max(1)[1].item()
		else:
			action = random.randrange(self.action_space)
		return action



class PongPolicy(nn.Module):
	def __init__(self, n):
		super(PongPolicy, self).__init__()
		self.fc1 = nn.Linear(6400, 200)
		self.fc2 = nn.Linear(200, 200)
		self.fc3 = nn.Linear(200, n)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		action_scores = self.fc3(x)
		return F.softmax(action_scores, dim=-1)
