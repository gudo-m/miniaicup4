import json
import random
import keyboard
import torch
from torch import nn
from torch import optim
import numpy as np
import datetime
from multiprocessing import Process

config = input()


class PolicyEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_inputs = 3
        self.n_outputs = 4

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs),
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


def make_ls(lstate):
    with open('state', 'w') as f:
        f.write(str(lstate))
        f.write('\n'+str(config))
    commands_int = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
    ls = lstate['params']['players']['i']
    direction = -1 if ls['direction'] is None else commands_int[ls['direction']]
    learning_state = {
        'direction': int(direction),
        'pos_x': int(ls['position'][0]),
        'pos_y': int(ls['position'][1])
    }
    reward = ls['score'] - score
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

if datetime.datetime.now().second % 15 == 0:
    torch.save(pe.state_dict(), PATH+f'-{datetime.datetime.now()}')
