import argparse
import random

import torch
import torch.nn as nn
from torch import optim

parser = argparse.ArgumentParser()

parser.add_argument('--dqn_hidden_dim', type=int, default=3)
parser.add_argument('--lr', type=float, default=.001)
parser.add_argument('--momentum', type=float, default=.7)
parser.add_argument('--gamma', type=float, default=.8)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--iter_before_training', type=int, default=200)
parser.add_argument('--eps', type=float, default=.1)
parser.add_argument('--memory_buffer_size', type=int, default=500)
parser.add_argument('--replace_target_every_n', type=int, default=30)
parser.add_argument('--log_every_n', type=int, default=50)
parser.add_argument('--num_train', type=int, default=3000)
parser.add_argument('--num_demo', type=int, default=1000)

args = vars(parser.parse_args())


def copy_nets(net1, net2):
    net1.load_state_dict(net2.state_dict())


def get_state():
    return torch.tensor([[random.randint(0, 1)]]).float()


ACTIONS = [torch.tensor([[1., 0.]]),
           torch.tensor([[0., 1.]])]


class DQNSolver:
    def __init__(self, q_net, target_q_net, optimizer,
                 gamma=.3, batch_size=8, iter_before_train=50, eps=.1,
                 memory_buffer_size=100, replace_target_every_n=100, log_every_n=100):
        self.q_net = q_net
        self.target_q_net = target_q_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.batch_size = batch_size
        self.iter_before_train = iter_before_train
        self.eps = eps
        self.memory_buffer_size = memory_buffer_size
        self.replace_target_every_n = replace_target_every_n
        self.log_every_n = log_every_n
        self.memory_buffer = []
        self.losses = []

    def get_q_value(self, state, action, use_target=True):
        q_net_input = torch.cat([state] + [action], 1)
        if use_target:
            return self.target_q_net(q_net_input)
        else:
            return self.q_net(q_net_input)

    @torch.no_grad()
    def select_action(self, state):
        if (len(self.memory_buffer) < self.iter_before_train or
                random.uniform(0, 1) < self.eps):
            return random.choice(ACTIONS)
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
            current_q = self.get_q_value(state, action.type(torch.FloatTensor), use_target=False)
            future_q_value = torch.tensor(0)
            for next_action in ACTIONS:
                candidate_q_value = self.get_q_value(state, next_action, use_target=False).detach()
                future_q_value = max(future_q_value, candidate_q_value)
            loss += (current_q - (reward + self.gamma * future_q_value)) ** 2

        loss.backward(retain_graph=True)
        optimizer.step()
        self.losses.append(loss.item())

    def train(self, num_iterations):
        triple = None
        for iteration in range(num_iterations):
            # Main iteration.
            state = get_state()
            action = self.select_action(state)
            if state[0, 0].item() == torch.argmax(action, dim=1):
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
        success_count = 0
        for _ in range(num_iterations):
            state = get_state()
            action = self.select_action(state)
            if state[0, 0].item() == torch.argmax(action, 1).item():
                success_count += 1
        return success_count / num_iterations


dqn_hidden_dim = args['dqn_hidden_dim']

dqn = nn.Sequential(
    nn.Linear(3, dqn_hidden_dim),
    nn.ReLU(),
    nn.Linear(dqn_hidden_dim, 1),
    nn.ReLU()
)
target_dqn = nn.Sequential(
    nn.Linear(3, dqn_hidden_dim),
    nn.ReLU(),
    nn.Linear(dqn_hidden_dim, 1),
    nn.ReLU()
)
target_dqn.load_state_dict(dqn.state_dict())
optimizer = optim.SGD(dqn.parameters(), lr=args['lr'], momentum=args['momentum'])

solver = DQNSolver(dqn,
                   target_dqn,
                   optimizer,
                   gamma=args['gamma'],
                   batch_size=args['batch_size'],
                   iter_before_train=args['iter_before_training'],
                   eps=args['eps'],
                   memory_buffer_size=args['memory_buffer_size'],
                   log_every_n=args['log_every_n'])

solver.train(args['num_train'])
success_rate = solver.eval(args['num_demo'])
print(f"\n\nSuccess Rate:  {success_rate}")
