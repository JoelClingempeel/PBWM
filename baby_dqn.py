import argparse
import random

import torch
import torch.nn as nn
from torch import optim

from baby_store_ignore_recall import GetData

parser = argparse.ArgumentParser()

parser.add_argument('--num_symbols', type=int, default=2)
parser.add_argument('--dqn_hidden_dim', type=int, default=3)
parser.add_argument('--lr', type=float, default=.0001)
parser.add_argument('--momentum', type=float, default=.7)
parser.add_argument('--gamma', type=float, default=.8)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--iter_before_training', type=int, default=200)
parser.add_argument('--eps', type=float, default=.3)
parser.add_argument('--memory_buffer_size', type=int, default=500)
parser.add_argument('--replace_target_every_n', type=int, default=30)
parser.add_argument('--log_every_n', type=int, default=100)
parser.add_argument('--num_train', type=int, default=3000)
parser.add_argument('--num_demo', type=int, default=50)
parser.add_argument('--ignore_prob', type=float, default=.5)
parser.add_argument('--interactive_mode', type=str, default='False')

args = vars(parser.parse_args())

INSTRUCTION_LIST = {0: 'Ignore', 1: 'Store', 2: 'Recall'}


def copy_nets(net1, net2):
    net1.load_state_dict(net2.state_dict())


def get_actions(length):
    out = []
    for index in range(2 ** length):
        vec = []
        for _ in range(length):
            vec.append(index % 2)
            index >>= 1
        out.append(vec)
    return out


ACTIONS = [torch.tensor([action]).float()
           for action in get_actions(2)]


class PFC:
    def __init__(self, stripe_size):
        self.stripe_size = stripe_size

        self.stripes = [torch.zeros(1, stripe_size)
                        for _ in range(2)]

    def update(self, data, gates):
        if gates[0, 0].item():
            self.stripes[0] = data
        if gates[0, 1].item():
            self.stripes[1] = self.stripes[0].detach()
        else:
            self.stripes[1] = torch.zeros(1, self.stripe_size)

    def output(self):
        for val in self.stripes[1].squeeze(0):
            if val.item():
                return torch.argmax(self.stripes[-1])
        return torch.tensor(0)


class DQNSolver:
    def __init__(self, data_src, q_net, target_q_net, pfc, optimizer, num_symbols,
                 gamma=.3, batch_size=8, iter_before_train=50, eps=.1,
                 memory_buffer_size=100, replace_target_every_n=100, log_every_n=100,
                 ignore_prob=.5, interactive_mode=False):
        self.data_src = data_src
        self.q_net = q_net
        self.target_q_net = target_q_net
        self.pfc = pfc
        self.optimizer = optimizer
        self.num_symbols = num_symbols
        self.gamma = gamma
        self.batch_size = batch_size
        self.iter_before_train = iter_before_train
        self.eps = eps
        self.memory_buffer_size = memory_buffer_size
        self.replace_target_every_n = replace_target_every_n
        self.log_every_n = log_every_n
        self.ignore_prob = ignore_prob
        self.interactive_mode = interactive_mode
        self.memory_buffer = []
        self.losses = []

    def get_q_value(self, state, action, use_target=False):
        q_net_input = torch.cat([state] + self.pfc.stripes + [action], 1)
        if use_target:
            return self.target_q_net(q_net_input)
        else:
            return self.q_net(q_net_input)

    @torch.no_grad()
    def select_action(self, state):
        if (len(self.memory_buffer) < self.iter_before_train or
                random.uniform(0, 1) < self.eps):
            return torch.tensor([[random.choice([0, 1])
                                  for _ in range(2)]])
        else:
            best_action_score = -1
            for action in ACTIONS:
                action_score = self.get_q_value(state, action)
                if action_score > best_action_score:
                    best = action
                    best_action_score = action_score
            return best

    def train_iterate(self):
        samples = random.sample(self.memory_buffer, self.batch_size)
        loss = 0
        optimizer.zero_grad()

        for state, action, reward, new_state in samples:
            current_q = self.get_q_value(state, action.type(torch.FloatTensor))
            future_q_value = torch.tensor(0)
            for next_action in ACTIONS:
                candidate_q_value = self.get_q_value(state, next_action, use_target=True).detach()
                future_q_value = max(future_q_value, candidate_q_value)
            loss += (current_q - (reward + self.gamma * future_q_value)) ** 2

        loss.backward(retain_graph=True)
        optimizer.step()
        self.losses.append(loss.item())

    def train(self, num_iterations):
        triple = None
        for iteration in range(num_iterations):
            # Main iteration.
            state, answer = self.data_src.get_data(ignore_prob=self.ignore_prob)
            action = self.select_action(state)
            self.pfc.update(state[:, :-3], action)
            if self.pfc.output() == answer:
                reward = 1
            else:
                reward = 0

            # Update memory buffer and (if applicable) train.
            if triple:
                self.memory_buffer.append(triple + [state])
            if len(self.memory_buffer) > self.memory_buffer_size:
                self.memory_buffer.pop(0)
            if len(self.memory_buffer) >= self.iter_before_train:
                self.train_iterate()

            triple = [state, action, reward]  # For next iteration

            if len(self.losses) == self.log_every_n:
                print(sum(self.losses) / len(self.losses))
                self.losses = []

            if (iteration + 1) % self.replace_target_every_n == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

    def eval(self, num_iterations):
        for _ in range(num_iterations):
            state, answer = self.data_src.get_data(ignore_prob=self.ignore_prob,
                                                   interactive=self.interactive_mode)
            symbol = torch.argmax(state[:, :self.num_symbols], 1).item() + 1
            print(f"Symbol:  {symbol}")
            instruction = INSTRUCTION_LIST[torch.argmax(state[:, self.num_symbols:], 1).item()]
            print(f"Action:  {instruction}")

            gating = self.select_action(state)
            print('GATING: ', gating)
            self.pfc.update(state[:, :-3], gating)
            print('GET: ', self.pfc.output().item())
            print('EXPECT: ', answer.item())
            print('\n\n')


num_symbols = args['num_symbols']
dqn_hidden_dim = args['dqn_hidden_dim']

data_src = GetData(num_symbols)
dqn = nn.Sequential(
    nn.Linear(num_symbols * 3 + 5, dqn_hidden_dim),
    nn.ReLU(),
    nn.Linear(dqn_hidden_dim, 1),
    nn.ReLU()
)
target_dqn = nn.Sequential(
    nn.Linear(num_symbols * 3 + 5, dqn_hidden_dim),
    nn.ReLU(),
    nn.Linear(dqn_hidden_dim, 1),
    nn.ReLU()
)
target_dqn.load_state_dict(dqn.state_dict())
pfc = PFC(num_symbols)
optimizer = optim.SGD(dqn.parameters(), lr=args['lr'], momentum=args['momentum'])

solver = DQNSolver(data_src,
                   dqn,
                   target_dqn,
                   pfc,
                   optimizer,
                   num_symbols,
                   gamma=args['gamma'],
                   batch_size=args['batch_size'],
                   iter_before_train=args['iter_before_training'],
                   eps=args['eps'],
                   memory_buffer_size=args['memory_buffer_size'],
                   log_every_n=args['log_every_n'],
                   ignore_prob=args['ignore_prob'],
                   interactive_mode=(args['interactive_mode'] == 'True'))

solver.train(args['num_train'])
solver.eps = 0
solver.eval(args['num_demo'])
