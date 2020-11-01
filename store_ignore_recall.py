import random

import torch

ACTIONS = {0: 'Ignore', 1: 'Store', 2: 'Recall'}


def one_hot(index, vec_size):
    vec = torch.zeros(1, vec_size)
    vec[0, index] = 1
    return vec


class GetData:
    def __init__(self, num_symbols, use_simplified_task=False):
        self.num_symbols = num_symbols
        self.stored_symbol = 0
        self.use_simplified_task = use_simplified_task

    def get_data_simplified(self, ignore_prob=.5, interactive=False):
        if self.stored_symbol == 0:  # Store or Ignore
            answer = 0
            if interactive:
                symbol = int(input(f'Please select a number between 1 and {self.num_symbols}.\n'))
                task = int(input('Please enter 0 for ignore and 1 for store.\n'))
            else:
                symbol = random.randint(1, self.num_symbols)
                rand_num = random.random()
                if rand_num < ignore_prob:
                    task = 0
                else:
                    task = 1
            if task == 1:  # Store
                self.stored_symbol = symbol
        else:  # Recall
            answer = self.stored_symbol
            symbol = random.randint(1, self.num_symbols)
            self.stored_symbol = 0
            task = 2

        symbol_vec = one_hot(symbol - 1, self.num_symbols)
        task_vec = one_hot(task, 3)
        return torch.cat([symbol_vec, task_vec], 1), torch.tensor(answer)

    def get_data(self, ignore_prob=.3, interactive=False):
        if self.use_simplified_task:
            return self.get_data_simplified(ignore_prob=ignore_prob, interactive=interactive)
        answer = 0
        if interactive:
            symbol = int(input(f'Please select a number between 1 and {self.num_symbols}.\n'))
            task = int(input('Please enter 0 for ignore, 1 for store, and (if a symbol is stored) 2 for recall.\n'))
        else:
            symbol = random.randint(1, self.num_symbols)
            rand_num = random.random()
            if rand_num < ignore_prob:
                task = 0
            else:
                task = random.randint(1, 2 if self.stored_symbol else 1)
        if task == 1:  # Store
            self.stored_symbol = symbol
        elif task == 2:  # Recall
            answer = self.stored_symbol

        symbol_vec = one_hot(symbol - 1, self.num_symbols)
        task_vec = one_hot(task, 3)
        return torch.cat([symbol_vec, task_vec], 1), torch.tensor(answer)

    def show_demo(self, state, out, label):
        symbol = torch.argmax(state[:, :self.num_symbols], 1).item() + 1
        print(f"Symbol:  {symbol}")
        action = ACTIONS[torch.argmax(state[:, self.num_symbols:], 1).item()]
        print(f"Action:  {action}")
        out = torch.argmax(out, 1).item()
        if out > 0:
            print(f"Bot recalls {out}.")
        else:
            print("Bot waits.")
        label = label[0].item()
        if label > 0:
            print(f"Correct:  Bot recalls {label}.")
        else:
            print("Correct:  Bot waits.")
        print("\n\n")
