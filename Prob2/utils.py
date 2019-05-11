import matplotlib
matplotlib.use('Agg')
import math, random
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import numpy as np

class replaybuffer(object):
    def __init__(self, buffer_length):
        self.buffer = deque(maxlen=buffer_length)

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))



def plot(frame_idx, rewards, losses, algo):
	print(frame_idx)
	plt.figure(figsize=(12,5))
	plt.subplot(121)
	plt.title('rewards')
	plt.plot(rewards)
	plt.subplot(122)
	plt.title('running reward')
	plt.plot(losses)
	plt.savefig('./' + str(algo) + str(frame_idx) + 'try.png')


class QNet(nn.Module):
	def __init__(self, input_size, action_space):
		super(QNet, self).__init__()

		self.input_size = input_size
		self.action_space = action_space

		self.relu = nn.ReLU()
		self.conv1 = nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

		self.fc1 = nn.Linear(3136, 512)
		self.fc2 = nn.Linear(512, self.action_space)

	def forward(self, x):

		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = x.view(x.size(0), -1)

		x = self.relu(self.fc1(x))
		x = self.fc2(x)
		return x

	def act(self, state, epsilon):
		if random.random() > epsilon:
			state   = torch.FloatTensor(np.float32(state)).unsqueeze(0).cuda()
			q_value = self.forward(state)
			action  = q_value.max(1)[1].data[0]
		else:
			action = random.randrange(self.action_space)
		return action
