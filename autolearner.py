import os
import time
import json
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import random


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


cmd = 'python3 localrunner.py -p3 "python3 strat.py"  -p1 simple_bot -p4 simple_bot -p5 simple_bot -p6 simple_bot -p2 simple_bot --no-gui'
cmd1 = 'python3 localrunner.py -p4 "python3 strat.py"  -p1 simple_bot -p3 simple_bot -p5 simple_bot -p6 simple_bot -p2 simple_bot --no-gui'
cmd3 = 'python3 localrunner.py -p5 "python3 strat.py"  -p1 simple_bot -p3 simple_bot -p4 simple_bot -p6 simple_bot -p2 simple_bot --no-gui'
cmd4 = 'python3 localrunner.py -p6 "python3 strat.py"  -p1 simple_bot -p3 simple_bot -p4 simple_bot -p5 simple_bot -p2 simple_bot --no-gui'
cmd5 = 'python3 localrunner.py -p2 "python3 strat.py"  -p1 simple_bot -p4 simple_bot -p5 simple_bot -p6 simple_bot -p3 simple_bot --no-gui'
cmd2 = 'python3 localrunner.py -p1 "python3 strat.py" -p2 simple_bot -p3 simple_bot -p4 simple_bot -p5 simple_bot -p6 simple_bot --no-gui'


def generate_batch(batch_size):
    print(f'generating batch...')
    batch_actions, batch_states, batch_rewards = [], [], []

    for b in range(batch_size):
        os.system(random.choice([cmd, cmd1, cmd2, cmd3, cmd4, cmd5]))
        time.sleep(0.1)
        os.system('find . -type f -name "*.log.gz" -exec rm -f {} \;')
        with open('log', 'r') as f:
            line = json.loads(f.readline().strip())
            batch_states.append(line)
            line = json.loads(f.readline().strip())
            batch_actions.append(line)
            line = json.loads(f.readline().strip())
            batch_rewards.append(line)
        print(f'{b+1}/{batch_size} of batch ended')
    return batch_states, batch_actions, batch_rewards


def filter_batch(states_batch, actions_batch, rewards_batch, percentile=50):
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_states = []
    elite_actions = []

    for i in range(len(rewards_batch)):
        if rewards_batch[i] >= reward_threshold:
            for j in range(len(states_batch[i])):
                elite_states.append(states_batch[i][j])
                elite_actions.append(actions_batch[i][j])

    return elite_states, elite_actions


if __name__ == '__main__':
    batch_size = 50
    session_size = 100
    percentile = 90
    hidden_size = 1024
    learning_rate = 0.0025
    completion_score = 200

    n_states = 963
    n_actions = 4

    # neural network
    net = Net(n_states, hidden_size, n_actions)
    # load params
    try:
        net.load_state_dict(torch.load('params/param19'))
        net.eval()
    except FileNotFoundError:
        pass
    # loss function
    objective = nn.CrossEntropyLoss()
    # optimisation function
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

    for i in range(session_size):
        # generate new sessions
        print(f'started session {i+1}')
        batch_states, batch_actions, batch_rewards = generate_batch(batch_size)
        elite_states, elite_actions = filter_batch(batch_states, batch_actions, batch_rewards, percentile)
        optimizer.zero_grad()
        tensor_states = torch.FloatTensor(elite_states)
        tensor_actions = torch.LongTensor(elite_actions)
        action_scores_v = net(tensor_states)
        loss_v = objective(action_scores_v, tensor_actions)
        loss_v.backward()
        optimizer.step()

        torch.save(net.state_dict(), 'params/param19')
        torch.save(net.state_dict(), 'params/param19' + f'-{datetime.datetime.now()}')

        # show results
        mean_reward = np.mean(batch_rewards),
        np.percentile(batch_rewards, percentile)
        print(f"{i}: loss={loss_v.item()}, reward_mean={mean_reward}")

