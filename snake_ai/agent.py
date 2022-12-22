#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the code for the agent.
"""


import random
import collections
from typing import List, Optional

import numpy as np
import torch

from .game import Direction, Action, Point, SnakeGameBot
from .helper import plot
from .model import Linear_QNet, QTrainer


def train(speed: int=60):
    """"""
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameBot(base_clock_speed=speed)

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

        # Remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        # 
        if game_over:
            # Train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            # Plot results
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)
            
            # Print the results
            print(f"Game {agent.n_games}")
            print(f"Score: {score}")
            print(f"Mean score: {mean_score}")
            print(f"Record: {record}")


class State:
    """"""

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
    } # TODO: Change to list

    SIZE = len(_INDEXES)

    def __init__(self, state: Optional[np.array]=None):
        """"""
        self._state = state if state is not None else np.zeros(11, dtype=int)
    
    def get(self) -> np.array:
        """"""
        return self._state
    
    def set(self, value):
        """"""
        self._state = value

    def __getitem__(self, item) -> int:
        """"""
        if isinstance(item, str):
            item = self._INDEXES[item]
        return self._state[item]
    
    def __setitem__(self, item, value):
        """"""
        if isinstance(item, str):
            item = self._INDEXES[item]
        self._state[item] = value
    
    def __repr__(self):
        """"""
        return f"State{np.array_str(self._state)}"
    
    def __str__(self):
        """"""
        return f"State{np.array_str(self._state)}"
    
    @property
    def danger_straight(self) -> int:
        """"""
        return self['danger straight']
    
    @danger_straight.setter
    def danger_straight(self, value):
        """"""
        self['danger straight'] = value
    
    @property
    def danger_right(self) -> int:
        """"""
        return self['danger right']
    
    @danger_right.setter
    def danger_right(self, value):
        """"""
        self['danger right'] = value
    
    @property
    def danger_left(self) -> int:
        """"""
        return self['danger left']
    
    @danger_left.setter
    def danger_left(self, value):
        """"""
        self['danger left'] = value
    
    @property
    def move_left(self) -> int:
        """"""
        return self['move left']
    
    @move_left.setter
    def move_left(self, value):
        """"""
        self['move left'] = value
    
    @property
    def move_right(self) -> int:
        """"""
        return self['move right']
    
    @move_right.setter
    def move_right(self, value):
        """"""
        self['move right'] = value
    
    @property
    def move_up(self) -> int:
        """"""
        return self['move up']
    
    @move_up.setter
    def move_up(self, value):
        """"""
        self['move up'] = value
    
    @property
    def move_down(self) -> int:
        """"""
        return self['move down']
    
    @move_down.setter
    def move_down(self, value):
        """"""
        self['move down'] = value
    
    @property
    def food_left(self) -> int:
        """"""
        return self['food left']
    
    @food_left.setter
    def food_left(self, value):
        """"""
        self['food left'] = value
    
    @property
    def food_right(self) -> int:
        """"""
        return self['food right']
    
    @food_right.setter
    def food_right(self, value):
        """"""
        self['food right'] = value
    
    @property
    def food_up(self) -> int:
        """"""
        return self['food up']
    
    @food_up.setter
    def food_up(self, value):
        """"""
        self['food up'] = value
    
    @property
    def food_down(self) -> int:
        """"""
        return self['food down']
    
    @food_down.setter
    def food_down(self, value):
        """"""
        self['food down'] = value


class Agent:
    """"""

    _MAX_MEMORY = 100_000
    _BATCH_SIZE = 1_000
    _LEARNING_RATE = 0.001

    _EPISILON_START = 80
    _HIDDEN_SIZE = 256

    def __init__(self):
        """"""
        self.n_games = 0
        self.epsilon = 0 # controls randomness
        self.gamma = 0.9 # discount rate, <1, normal ~0.8-0.9
        self.memory = collections.deque(maxlen=self._MAX_MEMORY)
        # self.model = Linear_QNet(11, 256, 3) # TODO: generalize this
        self.model = Linear_QNet(
            State.SIZE, self._HIDDEN_SIZE, len(Action)
        ) # TODO: generalize this
        self.trainer = QTrainer(
            self.model, learning_rate=self._LEARNING_RATE, gamma=self.gamma
        )

    def get_state(self, game: SnakeGameBot) -> State:
        """"""
        
        # head = game.snake[0]
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
        """"""
        self.memory.append((state, action, reward, next_state, finished))

    def train_long_memory(self):
        """"""
        
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
        """"""
        self.trainer.train_step(state, action, reward, next_state, finished)

    def get_action(self, state: State) -> Action:
        """"""

        # Perform random moves (tradeoff exploration/exploitation)
        self.epsilon = self._EPISILON_START - self.n_games
        if random.randint(0, 200) < self.epsilon:
            final_move = random.choice(list(Action))
        else:
            state_tensor = torch.tensor(state.get(), dtype=torch.float)
            prediction = self.model(state_tensor)
            move_index = torch.argmax(prediction).item()
            final_move = list(Action)[move_index]
        return final_move