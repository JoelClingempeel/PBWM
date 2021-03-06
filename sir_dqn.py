import argparse
import datetime
import os
import random

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from store_ignore_recall import GetData

parser = argparse.ArgumentParser()

parser.add_argument('--num_symbols', type=int, default=2)
parser.add_argument('--dqn_hidden_dim', type=int, default=8)
parser.add_argument('--lr', type=float, default=.001)
parser.add_argument('--momentum', type=float, default=.7)
parser.add_argument('--gamma', type=float, default=.95)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--iter_before_train', type=int, default=200)
parser.add_argument('--eps', type=float, default=.3)
parser.add_argument('--memory_buffer_size', type=int, default=500)
parser.add_argument('--replace_target_every_n', type=int, default=500)
parser.add_argument('--log_every_n', type=int, default=100)
parser.add_argument('--num_train', type=int, default=4000)
parser.add_argument('--num_demo', type=int, default=100)
parser.add_argument('--ignore_prob', type=float, default=0.33)
parser.add_argument('--use_simplified_task', type=str, default='False')
parser.add_argument('--interactive_mode', type=str, default='False')
parser.add_argument('--tensorboard_path', type=str, default='logs/tensorboard')
parser.add_argument('--examples_path', type=str, default='logs/examples')

args = vars(parser.parse_args())

INSTRUCTION_LIST = {0: 'Ignore', 1: 'Store', 2: 'Recall'}
TIMESTAMP = str(datetime.datetime.now()).replace(' ', '_')


def copy_nets(net1, net2):
    net1.load_state_dict(net2.state_dict())


class PFC:
    def __init__(self, stripe_size):
        self.stripe_size = stripe_size

        self.stripes = [torch.zeros(1, stripe_size)
                        for _ in range(2)]

    def update(self, data, action):
        if action in (1, 3):
            self.stripes[0] = data
        if action in (0, 2):
            self.stripes[1] = self.stripes[0].detach()
        else:
            self.stripes[1] = torch.zeros(1, self.stripe_size)

    def output(self):
        for val in self.stripes[1].squeeze(0):
            if val.item():
                return torch.argmax(self.stripes[-1]) + 1
        return torch.tensor(0)


class DQNSolver:
    def __init__(self, data_src, q_net, target_q_net, pfc, optimizer, num_symbols,
                 gamma=.3, batch_size=8, iter_before_train=50, eps=.1,
                 memory_buffer_size=100, replace_target_every_n=100, log_every_n=100,
                 ignore_prob=.5, interactive_mode=False, tensorboard_path='logs', examples_path=''):
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
        self.writer = SummaryWriter(os.path.join(tensorboard_path, TIMESTAMP))
        self.examples_path = os.path.join(examples_path, TIMESTAMP)
        self.memory_buffer = []
        self.losses = []
        self.rewards = []
        self.loss_log_count = 0
        self.reward_log_count = 0

    def get_q_values(self, state, use_target=False):
        q_net_input = torch.cat([state] + self.pfc.stripes, 1)
        if use_target:
            return self.target_q_net(q_net_input)
        else:
            return self.q_net(q_net_input)

    def select_action(self, state):
        if (len(self.memory_buffer) < self.iter_before_train or
                random.uniform(0, 1) < self.eps):
            return random.randint(0, 3)
        else:
            q_val = self.get_q_values(state)
            return torch.argmax(q_val.squeeze(0)).item()

    def train_iterate(self):
        samples = random.sample(self.memory_buffer, self.batch_size)
        loss = 0
        optimizer.zero_grad()

        for state, action, reward, new_state in samples:
            current_q = self.get_q_values(state).squeeze(0)[action]
            future_q_value = torch.max(self.get_q_values(state, use_target=True).detach().squeeze(0))
            loss += (current_q - (reward + self.gamma * future_q_value)) ** 2

        loss.backward(retain_graph=True)
        optimizer.step()
        self.losses.append(loss.item())

    def train(self, num_iterations):
        triple = None
        for iteration in range(num_iterations):
            # Main iteration.
            state, answer = self.data_src.get_data(ignore_prob=self.ignore_prob)
            with torch.no_grad():
                action = self.select_action(state)
            self.pfc.update(state[:, :self.num_symbols], action)
            if self.pfc.output() == answer:
                reward = 1
            else:
                reward = 0
            self.rewards.append(reward)

            # Update memory buffer and (if applicable) train.
            if triple:
                self.memory_buffer.append(triple + [state])
            if len(self.memory_buffer) > self.memory_buffer_size:
                self.memory_buffer.pop(0)
            if len(self.memory_buffer) >= self.iter_before_train:
                self.train_iterate()

            triple = [state, action, reward]  # For next iteration

            if len(self.losses) == self.log_every_n:
                self.writer.add_scalar('Loss', sum(self.losses) / len(self.losses), self.loss_log_count)
                self.losses = []
                self.writer.flush()
                self.loss_log_count += 1

            if len(self.rewards) == self.log_every_n:
                self.writer.add_scalar('Reward', sum(self.rewards) / len(self.rewards), self.reward_log_count)
                self.rewards = []
                self.writer.flush()
                self.reward_log_count += 1

            if (iteration + 1) % self.replace_target_every_n == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

    def eval(self, num_iterations):
        with torch.no_grad():
            with open(self.examples_path, 'w') as f:
                for _ in range(num_iterations):
                    state, answer = self.data_src.get_data(ignore_prob=self.ignore_prob,
                                                           interactive=self.interactive_mode)
                    symbol = torch.argmax(state[:, :self.num_symbols], 1).item() + 1
                    f.write(f'Symbol:  {symbol}\n')
                    instruction = INSTRUCTION_LIST[torch.argmax(state[:, self.num_symbols:], 1).item()]
                    f.write(f'Action:  {instruction}\n')

                    action = self.select_action(state)
                    f.write(f'ACTION: {action}\n')
                    self.pfc.update(state[:, :self.num_symbols], action)
                    f.write(f'GET: {self.pfc.output().item()}\n')
                    f.write(f'EXPECT: {answer.item()}\n')
                    if self.pfc.output() == answer:
                        f.write('REWARD: 1\n')
                    else:
                        f.write('REWARD: 0\n')
                    f.write(f'PFC: {self.pfc.stripes}\n\n')


num_symbols = args['num_symbols']
dqn_hidden_dim = args['dqn_hidden_dim']

data_src = GetData(num_symbols,
                   use_simplified_task=(args['use_simplified_task'] == 'True'))
dqn = nn.Sequential(
    nn.Linear(num_symbols * 3 + 3, dqn_hidden_dim),
    nn.ReLU(),
    nn.Linear(dqn_hidden_dim, 4),
)
target_dqn = nn.Sequential(
    nn.Linear(num_symbols * 3 + 3, dqn_hidden_dim),
    nn.ReLU(),
    nn.Linear(dqn_hidden_dim, 4),
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
                   iter_before_train=args['iter_before_train'],
                   eps=args['eps'],
                   memory_buffer_size=args['memory_buffer_size'],
                   log_every_n=args['log_every_n'],
                   ignore_prob=args['ignore_prob'],
                   interactive_mode=(args['interactive_mode'] == 'True'),
                   tensorboard_path=args['tensorboard_path'],
                   examples_path=args['examples_path'])

solver.eps = .9
solver.train(3000)
solver.eps = .7
solver.train(3000)
solver.eps = .5
solver.train(3000)
solver.eps = .3
solver.train(3000)
solver.eps = .1
solver.train(3000)
solver.eps = 0
solver.train(8000)
solver.eval(args['num_demo'])
