#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the code for the model.
"""


import os
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    """A linear Deep Q-Network model."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Initializes the Model instance."""

        # Initialize the parent class
        super().__init__()

        # Construct the layers
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, tensor: torch.tensor):
        """Compute output tensors from input tensors."""

        tensor = F.relu(self.linear_1(tensor))
        tensor = self.linear_2(tensor)
        return tensor


class Trainer:
    """The trainer object."""

    def __init__(self, model, learning_rate, gamma):
        """Initializes the Trainer instance."""

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
        """Performs a training step."""

        # Account for action being an Action instance or a list of them
        if isinstance(action, tuple):
            action = tuple([a.value for a in action])
        else:
            action = action.value
        # Account for action being a State instance or a list of them
        if isinstance(state, tuple):
            state = tuple([s.get() for s in state])
        else:
            state = state.get()
        # Account for action being a State instance or a list of them
        if isinstance(next_state, tuple):
            next_state = tuple([s.get() for s in next_state])
        else:
            next_state = next_state.get()
        
        # Convert to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # If only one dimension, then reshape
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            finished = (finished,)
        
        # First, predicted Q values with the current state
        Q = self.model(state)

        # Then, predict the next Q value
        target = Q.clone()
        for index in range(len(finished)):
            Q_new = reward[index]
            if not finished[index]:
                next_prediction = torch.max(self.model(next_state[index]))
                Q_new = reward[index] + self.gamma * next_prediction

            target[index][torch.argmax(action).item()] = Q_new
        
        # Utilize the optimizer
        self.optimizer.zero_grad()
        loss = self.criterion(target, Q)
        loss.backward()

        # Step the optimizer
        self.optimizer.step()