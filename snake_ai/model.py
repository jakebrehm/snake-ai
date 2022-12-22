#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the code for the model.
"""


from typing import List, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    """"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """"""
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.tensor):
        """"""
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, filename="model.pth"):
        """"""
        MODEL_DIRECTORY = "../data/model" # TODO: generalize this
        if not os.path.exists(MODEL_DIRECTORY):
            os.makedirs(MODEL_DIRECTORY)
        
        filepath = os.path.join(MODEL_DIRECTORY, filename)
        torch.save(self.state_dict(), filepath)


class QTrainer:
    """"""

    def __init__(self, model, learning_rate, gamma):
        """"""
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_step(self,
        state: Union[List['State'], 'State'],
        action: Union[List['Action'], 'Action'],
        reward: Union[List[int], int],
        next_state: Union[List['State'], 'State'],
        finished: Union[List[bool], bool],
    ):
        """"""

        # print('\n')

        # print(f'{type(action)=}')
        # print(f'{action=}')
        # print(f'{action.value=}')

        if isinstance(action, tuple):
            action = tuple([a.value for a in action])
        else:
            action = action.value
        
        if isinstance(state, tuple):
            state = tuple([s.get() for s in state])
        else:
            state = state.get()
        
        if isinstance(next_state, tuple):
            next_state = tuple([s.get() for s in next_state])
        else:
            next_state = next_state.get()
        
        # print(action)

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        # state = torch.tensor(state.get(), dtype=torch.float)
        # next_state = torch.tensor(next_state.get(), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # If only one dimension, then reshape
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            finished = (finished,)
        
        # First, predicted Q values with the current state
        Q = self.model(state)

        target = Q.clone()
        for index in range(len(finished)):
            Q_new = reward[index]
            if not finished[index]:
                next_prediction = torch.max(self.model(next_state[index]))
                Q_new = reward[index] + self.gamma * next_prediction

            target[index][torch.argmax(action).item()] = Q_new
        
        # 
        self.optimizer.zero_grad()
        loss = self.criterion(target, Q)
        loss.backward()

        self.optimizer.step()