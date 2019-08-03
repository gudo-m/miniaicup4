import json
import random
import keyboard
import torch
from torch import nn
from torch import optim
import numpy as np
import datetime
from multiprocessing import Process

config = json.loads(input())

flag = True


class PolicyEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_inputs = 964
        self.n_outputs = 4

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_outputs),
            nn.Softmax(dim=-1)
        )

    def predict(self, now_state):
        action_props = self.network(torch.FloatTensor(now_state))
        return action_props


def discount_rewards(lrewards, gamma=0.99):
    r = np.array([gamma**i * lrewards[i]
                  for i in range(len(lrewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def make_normal_coord(coord):
    return coord // config['params']['width']


def edit_cells_by_player(cells, player):
    for my_cell in player['territory']:
        x = make_normal_coord(my_cell[0])
        y = make_normal_coord(my_cell[1])
        cells[x * y] = 2
    if player.get('lines', False):
        for line in player['lines']:
            x = make_normal_coord(line[0])
            y = make_normal_coord(line[1])
            cells[x * y] = 3
    x = make_normal_coord(player['position'][0])
    y = make_normal_coord(player['position'][1])
    cells[x * y] = -1
    return cells


def make_ls(lstate):

    commands_int = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
    ls = lstate['params']['players']['i']

    direction = -1 if ls['direction'] is None else commands_int[ls['direction']]
    cells = [1 for _ in range(31*31)]

    for my_cell in ls['territory']:
        x = make_normal_coord(my_cell[0])
        y = make_normal_coord(my_cell[1])
        cells[x * y] = 0
    if ls.get('lines', False):
        for line in ls['lines']:
            x = make_normal_coord(line[0])
            y = make_normal_coord(line[1])
            cells[x * y] = -2
    x = make_normal_coord(ls['position'][0])
    y = make_normal_coord(ls['position'][1])
    cells[x * y] = -1

    learning_state = {
        'direction': int(direction),
        'pos_x': int(ls['position'][0]),
        'pos_y': int(ls['position'][1]),
    }

    for cell_i in range(len(cells)):
        learning_state[f'cell{cell_i}'] = cells[cell_i]

    reward = ls['score'] - (score)

    with open('state', 'w') as f:
        f.write(str(lstate))
        f.write('\n'+str(config))
        f.write('\n'+str(learning_state))
    return learning_state, reward


PATH = 'params/param19'
score = 0

total_rewards = []
batch_rewards = []
batch_actions = []
batch_states = []
batch_counter = 1

batch_size = 10

states = []
rewards = []
actions = []

pe = PolicyEstimator()
optimizer = optim.Adam(pe.network.parameters(), lr=0.01)

pe.load_state_dict(torch.load(PATH))
pe.eval()


while True:
    commands_int = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
    commands = ['left', 'right', 'up', 'down']

    try:
        state = json.loads(input())
    except:
        state = {'type': 'end_game'}

    if state['type'] == 'end_game':
        batch_rewards.extend(discount_rewards(rewards, 0.99))
        batch_states.extend(states)
        batch_actions.extend(actions)
        batch_counter += 1
        total_rewards.append(sum(rewards))

        # # If batch is complete, update network
        # if batch_counter == batch_size:
        optimizer.zero_grad()

        state_tensor = torch.FloatTensor(batch_states)
        reward_tensor = torch.FloatTensor(batch_rewards)
        # Actions are used as indices, must be LongTensor
        action_tensor = torch.LongTensor(batch_actions)

        # Calculate loss
        logprob = torch.log(
            pe.predict(state_tensor))
        selected_logprobs = reward_tensor * \
                            logprob[np.arange(len(action_tensor)), action_tensor]
        loss = -selected_logprobs.mean()

        # Calculate gradients
        loss.backward()
        # Apply gradients
        optimizer.step()

        with open('log', 'a+') as f:
            f.writelines(
                [
                    f"\r\rAverage of last 10: {np.mean(total_rewards[-10:])}",
                    f"\nDate: {datetime.datetime.now()}",
                ]
            )

        if np.mean(total_rewards[-10:]) > 0 and flag:
            torch.save(pe.state_dict(), PATH + f'(больше 0)-{datetime.datetime.now()}')
            flag = False
        elif not flag:
            flag = True

        break

    ls, reward = make_ls(state)

    action_probs = pe.predict(list(ls.values())).detach().numpy()

    action = np.random.choice([0,1,2,3], p=action_probs)
    cmd = commands[action]

    states.append(list(ls.values()))
    rewards.append(reward)
    actions.append(action)

    print(json.dumps({"command": cmd, 'debug': str(cmd)}))

torch.save(pe.state_dict(), PATH)

if datetime.datetime.now().second % 15 in (0, 5, 3):
    torch.save(pe.state_dict(), PATH+f'-{datetime.datetime.now()}')
