import json
import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from autolearner import Net
config = json.loads(input())
score = 0


def make_first_ls(ls):
    learning_state = {
        'direction': int(ls['direction']),
        'pos_x': int(ls['pos_x']),
        'pos_y': int(ls['pos_y']),
    }
    cells = [1 for _ in range(31*31)]
    for cell_i in range(len(cells)):
        learning_state[f'cell{cell_i}'] = cells[cell_i]
    return learning_state


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

    reward = ls['score'] - (score + 1 - (lstate['params'].get('tick_num', 0)/1500))
    return learning_state, reward


model = Net(964, 128, 4)
activation = nn.Softmax(dim=0)
states, actions = [], []
total_reward = 0


def get_command(state):
    ls, reward = make_ls(state)
    ls = list(ls.values())

    action = int(torch.argmax(model.predict(ls)))

    return action


s = None

while True:
    try:
        new_s = json.loads(input())
        new_s, r = make_ls(new_s)
        new_s = new_s
    except:
        break

    if s is None:
        s = list(make_first_ls(new_s).values())
    new_s = list(new_s.values())

    commands = ['left', 'right', 'up', 'down']

    s_v = torch.FloatTensor(s)
    act_probs_v = activation(model(s_v))
    act_probs = act_probs_v.data.numpy()
    a = np.random.choice(commands, p=act_probs_v.detach().numpy())
    states.append(s)
    actions.append(commands.index(a))
    total_reward += r

    s = new_s

    print(json.dumps({"command": a, 'debug': str(new_s)}))

with open('log', 'w') as f:
    f.write(
        str(states)+'\n'
        + str(actions)+'\n'
        + str(total_reward)
    )
