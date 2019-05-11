import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(ActorCritic, self).__init__()

        self.tanh = nn.Tanh()

        self.act_lin1 = nn.Linear(num_inputs, hidden_size)
        self.act_lin2 = nn.Linear(hidden_size, num_actions)
        self.softmax = nn.Softmax(dim=1)

        self.critic_lin1 = nn.Linear(num_inputs, hidden_size)
        self.critic_lin2 = nn.Linear(hidden_size, num_actions)


    def forward(self, x):

        # print(x.size())
        policy = self.softmax(self.act_lin2(self.tanh(self.act_lin1(x)))).clamp(max=1-1e-20)
        q_value = self.critic_lin2(self.tanh(self.critic_lin1(x)))

        value = (policy * q_value).sum(-1, keepdim=True)
        return policy, q_value, value
