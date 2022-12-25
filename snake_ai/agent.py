#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the code for the agent.
"""


import collections
import random
from typing import List, Optional

import numpy as np
import torch

from . import plotting
from .game import Action, Direction, Point, SnakeGameBot
from .model import Model, Trainer


def train(speed: int=60):
    """Trains the Snake bot."""

    # Initialize training variables
    scores = []
    mean_scores = []
    total_score = 0
    record = 0

    # Initialize the agent and the bot
    agent = Agent()
    game = SnakeGameBot(base_clock_speed=speed)

    # Initialize the ability to plot
    plotting.enable()

    # Start the training loop
    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, game_over, score = game.step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(
            state_old, final_move, reward, state_new, game_over
        )

        # Keep the values in the agent's memory
        agent.remember(state_old, final_move, reward, state_new, game_over)

        # Whenever a game is lost, analyze the results
        if game_over:
            # Train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # If the snake achieves a new record, store it
            if score > record:
                record = score

            # Plot results
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            plotting.add(scores, mean_scores)
            
            # Print the results
            print(f"Game {agent.n_games}")
            print(f"Score: {score}")
            print(f"Mean score: {mean_score}")
            print(f"Record: {record}")


class State:
    """Holds information about the game state.
    
    This information is used as the input neurons of the neural network.
    """

    _INDEXES = {
        'danger straight': 0,
        'danger right': 1,
        'danger left': 2,
        'move left': 3,
        'move right': 4,
        'move up': 5,
        'move down': 6,
        'food left': 7,
        'food right': 8,
        'food up': 9,
        'food down': 10,
    }

    SIZE = len(_INDEXES)

    def __init__(self, state: Optional[np.array]=None):
        """Initializes the State instance."""
        self._state = state if state is not None else np.zeros(11, dtype=int)
    
    def get(self) -> np.array:
        """Gets the state."""
        return self._state
    
    def set(self, value):
        """Sets the state."""
        self._state = value

    def __getitem__(self, item) -> int:
        """Gets a specific value of the state."""
        if isinstance(item, str):
            item = self._INDEXES[item]
        return self._state[item]
    
    def __setitem__(self, item, value):
        """Sets a specific value of the state."""
        if isinstance(item, str):
            item = self._INDEXES[item]
        self._state[item] = value
    
    def __repr__(self):
        """Returns the representation of the state."""
        return f"State{np.array_str(self._state)}"
    
    def __str__(self):
        """Returns the string representation of the state."""
        return f"State{np.array_str(self._state)}"
    
    @property
    def danger_straight(self) -> int:
        """Gets the value of 'danger straight'."""
        return self['danger straight']
    
    @danger_straight.setter
    def danger_straight(self, value):
        """Sets the value of 'danger straight'."""
        self['danger straight'] = value
    
    @property
    def danger_right(self) -> int:
        """Gets the value of 'danger right'."""
        return self['danger right']
    
    @danger_right.setter
    def danger_right(self, value):
        """Sets the value of 'danger right'."""
        self['danger right'] = value
    
    @property
    def danger_left(self) -> int:
        """Gets the value of 'danger left'."""
        return self['danger left']
    
    @danger_left.setter
    def danger_left(self, value):
        """Sets the value of 'danger left'."""
        self['danger left'] = value
    
    @property
    def move_left(self) -> int:
        """"Gets the value of 'move left'."""
        return self['move left']
    
    @move_left.setter
    def move_left(self, value):
        """Sets the value of 'move left'."""
        self['move left'] = value
    
    @property
    def move_right(self) -> int:
        """Gets the value of 'move right'."""
        return self['move right']
    
    @move_right.setter
    def move_right(self, value):
        """Gets the value of 'move right'."""
        self['move right'] = value
    
    @property
    def move_up(self) -> int:
        """Gets the value of 'move up'."""
        return self['move up']
    
    @move_up.setter
    def move_up(self, value):
        """Sets the value of 'move up'."""
        self['move up'] = value
    
    @property
    def move_down(self) -> int:
        """Gets the value of 'move down'."""
        return self['move down']
    
    @move_down.setter
    def move_down(self, value):
        """Sets the value of 'move down'."""
        self['move down'] = value
    
    @property
    def food_left(self) -> int:
        """Gets the value of 'food left'."""
        return self['food left']
    
    @food_left.setter
    def food_left(self, value):
        """Sets the value of 'food left'."""
        self['food left'] = value
    
    @property
    def food_right(self) -> int:
        """Gets the value of 'food right'."""
        return self['food right']
    
    @food_right.setter
    def food_right(self, value):
        """Sets the value of 'food right'."""
        self['food right'] = value
    
    @property
    def food_up(self) -> int:
        """Gets the value of 'food up'."""
        return self['food up']
    
    @food_up.setter
    def food_up(self, value):
        """Sets the value of 'food up'."""
        self['food up'] = value
    
    @property
    def food_down(self) -> int:
        """Gets the value of 'food down'."""
        return self['food down']
    
    @food_down.setter
    def food_down(self, value):
        """Sets the value of 'food down'."""
        self['food down'] = value


class Agent:
    """Agent that interacts with the game."""

    _MAX_MEMORY = 100_000
    _BATCH_SIZE = 1_000
    _LEARNING_RATE = 0.001

    _EPISILON_START = 80
    _HIDDEN_SIZE = 256

    def __init__(self, hidden_size: Optional[int]=None):
        """Initializes the Agent instance."""

        # If size of the hidden layers is specified, do not use default
        if hidden_size is not None:
            self._HIDDEN_SIZE = hidden_size

        # Track the number of games
        self.n_games = 0

        # Training parameters
        self.epsilon = 0 # controls randomness
        self.gamma = 0.9 # discount rate, <1, normal ~0.8-0.9

        # Initialize the agent's memory, the model, and the trainer
        self.memory = collections.deque(maxlen=self._MAX_MEMORY)
        self.model = Model(
            State.SIZE, self._HIDDEN_SIZE, len(Action)
        )
        self.trainer = Trainer(
            self.model, learning_rate=self._LEARNING_RATE, gamma=self.gamma
        )

    def get_state(self, game: SnakeGameBot) -> State:
        """Gets the current state of the game."""
        
        head = game.head

        # 
        point_left = Point(head.x - game.block_size, head.y)
        point_right = Point(head.x + game.block_size, head.y)
        point_up = Point(head.x, head.y - game.block_size)
        point_down = Point(head.x, head.y + game.block_size)

        # Current direction
        direction_left = (game.direction == Direction.LEFT)
        direction_right = (game.direction == Direction.RIGHT)
        direction_up = (game.direction == Direction.UP)
        direction_down = (game.direction == Direction.DOWN)

        # Create the state object
        state = State()

        state.danger_straight = (
            (direction_right and game.check_for_collision(point_right)) or
            (direction_left and game.check_for_collision(point_left)) or
            (direction_up and game.check_for_collision(point_up)) or
            (direction_down and game.check_for_collision(point_down))
        )

        state.danger_right = (
            (direction_up and game.check_for_collision(point_right)) or
            (direction_down and game.check_for_collision(point_left)) or
            (direction_left and game.check_for_collision(point_up)) or
            (direction_right and game.check_for_collision(point_down))
        )

        state.danger_left = (
            (direction_down and game.check_for_collision(point_right)) or
            (direction_up and game.check_for_collision(point_left)) or
            (direction_right and game.check_for_collision(point_up)) or
            (direction_left and game.check_for_collision(point_down))
        )

        state.move_left = direction_left
        state.move_right = direction_right
        state.move_up = direction_up
        state.move_down = direction_down

        state.food_left = game.food.x < game.head.x
        state.food_right = game.food.x > game.head.x
        state.food_up = game.food.y < game.head.y
        state.food_down = game.food.y > game.head.y

        # Return the state object
        return state

    def remember(self,
        state: State,
        action: Action,
        reward: int,
        next_state: State,
        finished: bool,
    ):
        """Store the given values in memory as a tuple."""
        self.memory.append((state, action, reward, next_state, finished))

    def train_long_memory(self):
        """Trains the long term memory."""
        
        if len(self.memory) > self._BATCH_SIZE:
            batch = random.sample(self.memory, self._BATCH_SIZE)
        else:
            batch = self.memory
        
        states, actions, rewards, next_states, finisheds = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, finisheds)

    def train_short_memory(self,
        state: State,
        action: Action,
        reward: int,
        next_state: State,
        finished: bool,
    ):
        """Trains the short term memory."""
        self.trainer.train_step(state, action, reward, next_state, finished)

    def get_action(self, state: State) -> Action:
        """Determine the next action of the snake."""

        # Perform random moves (tradeoff exploration/exploitation)
        self.epsilon = self._EPISILON_START - self.n_games
        if random.randint(0, 200) < self.epsilon:
            action = random.choice(list(Action))
        else:
            state_tensor = torch.tensor(state.get(), dtype=torch.float)
            prediction = self.model(state_tensor)
            move_index = torch.argmax(prediction).item()
            action = list(Action)[move_index]
        # Return the action
        return action