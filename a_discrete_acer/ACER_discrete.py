import gym
import numpy as np
import torch
import torch.optim as optim
import argparse
from network import ActorCritic
from utils import plot, ReplayBuffer, entropy_calc
from trpo import TRPO

parser = argparse.ArgumentParser(description=None)

parser.add_argument('--seed', default=123, type=int, help='random seed')
parser.add_argument('--batch_size', default=32, type=int, help='random seed')
parser.add_argument('--max_frames', default=10000, type=int, help='num frames')
parser.add_argument('--num_steps', default=5, type=int, help='number of steps')
parser.add_argument('--capacity', default=1000000, type=int, help='capacity')
parser.add_argument('--log_dur', default=100, type=int, help='log-interval')
parser.add_argument('--replay_ratio', default=8, type=int)
parser.add_argument('--truncation_clip', default=10, type=int)
parser.add_argument('--max_episode_length', default=200, type=int, help='max_episode_length')
parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
parser.add_argument('--entropy_weight', default=0.0001, type=float)
parser.add_argument('--alpha', default=0.95, type=float)
parser.add_argument('--type', default='notrpo', type=str, help='iftrpo')
parser.add_argument('--render', action='store_true', help='render')

args = parser.parse_args()
# print(args)
torch.manual_seed(args.seed)

env = gym.make("CartPole-v0")
replay_buffer = ReplayBuffer(args.capacity, args.max_episode_length)
model = ActorCritic(env.observation_space.shape[0], env.action_space.n).cuda()
average_model = ActorCritic(env.observation_space.shape[0], env.action_space.n).cuda()
optimizer = optim.Adam(model.parameters())

frame_idx = 0
test_rewards = []
episode_count = 0
step_count = 0
state = env.reset()

running_rew = 0
plotcount = 0

while frame_idx < args.max_frames:

    policies = []
    average_policies = []
    actions  = []
    rewards  = []
    masks    = []
    q_values = []
    values   = []

    for step in range(args.num_steps):

        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        policy, q_value, value = model(state)
        # print(policy, q_value)

        action = policy.multinomial(1)
        next_state, reward, done, _ = env.step(action.item())
        step_count += 1

        reward = torch.FloatTensor([reward]).unsqueeze(1).cuda()
        mask = torch.FloatTensor(1 - np.float32([done])).unsqueeze(1).cuda()
        replay_buffer.push(state.detach(), action, reward, policy.detach(), mask, done)

        policies.append(policy)
        actions.append(action)
        rewards.append(reward)
        masks.append(mask)
        q_values.append(q_value)
        values.append(value)

        state = next_state
        if done:
            state = env.reset()
            episode_count += 1


    next_state = torch.FloatTensor(state).unsqueeze(0).cuda()
    gaga, lala, retrace = model(next_state)

    # if retrace == torch.tensor(float('nan')):
    #     print("help")

    retrace = retrace.detach()

    behavior_policies = policies
    loss = 0

    for step in reversed(range(len(rewards))):

        imp_wt = policies[step].detach() / behavior_policies[step].detach()

        retrace = rewards[step] + args.gamma * retrace * masks[step]
        advantage = retrace - values[step]

        log_policy_action = policies[step].gather(1, actions[step]).log()
        trunc_imp_wt = imp_wt.gather(1, actions[step]).clamp(max=args.truncation_clip)
        actor_loss = -1 * (trunc_imp_wt * log_policy_action * advantage.detach()).mean(0)

        # if actor_loss == torch.tensor(float('nan')):
        #     print("help2")

        correction_weight = (1 - args.truncation_clip / imp_wt).clamp(min=0)
        actor_loss -= (correction_weight * policies[step].log() * (q_values[step] - values[step]).detach()).sum(1).mean(0)

        entropy = args.entropy_weight * entropy_calc(policies[step])

        q_value = q_values[step].gather(1, actions[step])
        critic_loss = ((retrace - q_value) ** 2 / 2).mean(0)

        truncated_rho = imp_wt.gather(1, actions[step]).clamp(max=1)

        # print(truncated_rho, critic_loss)

        retrace = truncated_rho * (retrace - q_value.detach()) + values[step].detach()

        loss += actor_loss + critic_loss - entropy

        if args.type == 'trpo':
            loss = TRPO(model, policies, average_policies, 1, loss, policies[step] / average_policies[step])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if args.batch_size < len(replay_buffer) + 1:
        for _ in range(np.random.poisson(args.replay_ratio)):
            trajecs = replay_buffer.sample(args.batch_size)
            s_x, a_x, r_x, old_pol, m_x = map(torch.stack, zip(*(map(torch.cat, zip(*trajec)) for trajec in trajecs)))

            q_vals = []
            vals   = []
            pols = []
            avg_pols = []

            for step in range(s_x.size(0)):
                pol, q_val, val = model(s_x[step])
                q_vals.append(q_val)
                pols.append(pol)
                vals.append(val)

            taga, saga, retr = model(s_x[-1])
            retr = retr.detach()

            loss = 0

            for step in reversed(range(len(r_x))):

                imp_wt = pols[step].detach() / old_pol[step].detach()

                #retrace is fine, gradients not exploding
                retr = r_x[step] + args.gamma * retr * m_x[step]
                advantage = retr - vals[step]

                log_policy_action = pols[step].gather(1, a_x[step]).log()
                #clamp with truncation clip
                trunc_imp_wt = imp_wt.gather(1, a_x[step]).clamp(max=args.truncation_clip)
                actor_loss = -(trunc_imp_wt * log_policy_action * advantage.detach()).mean(0)

                correction_weight = (1 - args.truncation_clip / imp_wt).clamp(min=0)
                actor_loss -= (correction_weight * pols[step].log() * (q_vals[step] - vals[step]).detach()).sum(1).mean(0)

                # print(actor_loss)
                # if actor_loss == torch.tensor(float('nan')):
                #     print("help")

                q_value = q_vals[step].gather(1, a_x[step])
                critic_loss = ((retr - q_value) ** 2 / 2).mean(0)

                entropy = args.entropy_weight * entropy_calc(pols[step])

                truncated_rho = imp_wt.gather(1, a_x[step]).clamp(max=1)
                retr = truncated_rho * (retr - q_value.detach()) + vals[step].detach()

                loss += actor_loss + critic_loss - entropy

                if args.type == 'trpo':
                    loss = TRPO(model, policies, average_policies, 1, loss, policies[step] / average_policies[step])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    if frame_idx % args.log_dur == 0:
        plotcount += 1
        print("Episode count", episode_count, "Buffer len", len(replay_buffer), "Step_count", step_count)

        vals = []
        for i in range(5):
            state = env.reset()
            if args.render and frame_idx % 2000 == 0:
                env.render()
            done = False
            total_reward = 0
            while not done:
                state = torch.FloatTensor(state).unsqueeze(0).cuda()
                policy, haga, zaga = model(state)
                action = policy.multinomial(1)
                # okay now we are getting diff actions
                # print(action)
                next_state, reward, done, yaya = env.step(action.item())
                if args.render and frame_idx % 2000 == 0:
                    env.render()
                state = next_state
                total_reward += reward

            state = env.reset()
            if args.render and frame_idx % 2000 == 0:
                env.render()
            vals.append(total_reward)
        val = np.mean(vals)

        print("Rewards: ", val)
        test_rewards.append(val)
        plot(frame_idx, test_rewards)

    frame_idx += args.num_steps
