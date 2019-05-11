from collections import deque
import random
import matplotlib.pyplot as plt

def plot(frame_idx, rewards):
    plt.title('ACER Cartpole Rewards')
    plt.xlabel('Steps * 100')
    plt.plot(rewards)
    plt.savefig('./' +  str(frame_idx) + 'run.png')

class ReplayBuffer(object):
    def __init__(self, capacity, max_episode_length):
        self.num_eps = capacity // max_episode_length
        print(self.num_eps)
        self.buffer = deque(maxlen=self.num_eps)
        self.buffer.append([])
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        smallest = 0
        while smallest == 0:
            rand_episodes = random.sample(self.buffer, batch_size - 2)
            smallest = min(len(episode) for episode in rand_episodes)

        # print(smallest)
        end_length = smallest
        episodes = []
        for episode in rand_episodes:
            if len(episode) > end_length:
                rand_idx = random.randint(0, len(episode) - end_length)
                # print(rand_idx)
            else:
                rand_idx = 0

            episodes.append(episode[rand_idx: rand_idx + end_length])

        return list(map(list, zip(*episodes)))

    def push(self, state, action, reward, policy, mask, done):
        self.buffer[self.position].append((state, action, reward, policy, mask))
        if done:
            # print(self.position)
            self.buffer.append([])
            self.position = min(self.position + 1, self.num_eps - 1)


def entropy_calc(policy_step):

    entropy = -1 * (policy_step.log() * policy_step).sum(1).mean(0)
    return entropy
