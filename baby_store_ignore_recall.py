import random

import torch
import torch.nn as nn

ACTIONS = {0: 'Ignore', 1: 'Store', 2: 'Recall'}


def one_hot(index, vec_size):
    vec = torch.zeros(1, vec_size)
    if index >= 0:
        vec[0, index] = 1
    return vec


class GetData:
    def __init__(self, num_symbols):
        self.num_symbols = num_symbols
        self.stored_symbol = 0

    def get_data(self, interactive=False):
        if self.stored_symbol == 0:  # Store or Ignore
            answer = 0
            if interactive:
                symbol = int(input('Please select a number between 1 and n.\n'))
                task = int(input('Please enter 0 for ignore and 1 for store.\n'))
            else:
                symbol = random.randint(1, self.num_symbols)
                task = random.randint(0, 1)
            if task == 1:  # Store
                self.stored_symbol = symbol
        else:  # Recall
            answer = self.stored_symbol
            symbol = 0
            self.stored_symbol = 0
            task = 2

        symbol_vec = one_hot(symbol - 1, self.num_symbols)  # For Recall gives zero vector.
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
