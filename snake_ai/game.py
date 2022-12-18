#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the code for the Snake game.
"""


import os
import random
from collections import namedtuple
from enum import Enum
from typing import List

import pygame


# Initialize PyGame
pygame.init()


# Get the path to the data directory
PARENT_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
DATA_DIRECTORY = os.path.join(PARENT_DIRECTORY, 'data')

# Get the path to import files in the data directory
FONT_PATH = os.path.join(DATA_DIRECTORY, 'Roboto-Regular.ttf')

# Define the game constants
SEGMENT_SIZE = 20
CLOCK_SPEED = 20 # make size of snake dependent on speed

# Define the colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE_1 = (0, 0, 255)
BLUE_2 = (0, 100, 255)

# Define the fonts
roboto = pygame.font.Font(FONT_PATH, 25)

# Define the namedtuple which stores point information
Point = namedtuple('Point', 'x, y')


class Direction(Enum):
    """"""

    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeGame:
    """"""

    _GAME_TITLE = "Snake"

    def __init__(self, width: int=640, height: int=480):
        """"""
        
        # Store user inputs
        self._width = width
        self._height = height

        # Initialize display
        self._display = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption(self._GAME_TITLE)
        self._clock = pygame.time.Clock()

        # Initialize game state
        self._direction = Direction.RIGHT
        self._head = Point(self._width/2, self._height/2)
        self._snake = [
            Point(self.head.x-(0*SEGMENT_SIZE), self.head.y),
            Point(self.head.x-(1*SEGMENT_SIZE), self.head.y),
            Point(self.head.x-(2*SEGMENT_SIZE), self.head.y),
        ]

        self._score = 0
        self._food = None
        self._place_food()
    
    def _place_food(self):
        """"""
        x = random.randint(0, (self._width-SEGMENT_SIZE)//SEGMENT_SIZE) * SEGMENT_SIZE
        y = random.randint(0, (self._height-SEGMENT_SIZE)//SEGMENT_SIZE) * SEGMENT_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _update_ui(self):
        """"""

        self.display.fill(BLACK)

        for point in self.snake:
            pygame.draw.rect(self.display, BLUE_1, pygame.Rect(point.x, point.y, SEGMENT_SIZE, SEGMENT_SIZE))
            pygame.draw.rect(self.display, BLUE_2, pygame.Rect(point.x+4, point.y+4, 12, 12))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, SEGMENT_SIZE, SEGMENT_SIZE))
        
        text = roboto.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move_snake(self, direction: Direction):
        """"""
        
        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += SEGMENT_SIZE
        elif direction == Direction.LEFT:
            x -= SEGMENT_SIZE
        elif direction == Direction.DOWN:
            y += SEGMENT_SIZE
        elif direction == Direction.UP:
            y -= SEGMENT_SIZE
        
        self.head = Point(x, y)

    def _check_for_collision(self):
        """"""
        # Check if snake hits the boundary
        if (self.head.x > (self._width - SEGMENT_SIZE)) or (self.head.x < 0):
            return True
        if (self.head.y > (self._height - SEGMENT_SIZE)) or (self.head.y < 0):
            return True

        # Check if snake hits itself
        if self.head in self.snake[1:]:
            return True
        
        # Otherwise, there was no collision
        return False

    def play_step(self):
        """"""

        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        # Move snake
        self._move_snake(self.direction) # update the head
        self.snake.insert(0, self.head)

        # Check if game over
        game_over = False
        if self._check_for_collision():
            game_over = True
            return game_over, self.score

        # Place new food or finalize move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # Update the pygame UI and the clock
        self._update_ui()
        self.clock.tick(CLOCK_SPEED)

        # Return game over and score
        return game_over, self.score
    
    @property
    def score(self) -> int:
        """"""
        return self._score
    
    @score.setter
    def score(self, value):
        """"""
        self._score = value

    @property
    def display(self) -> pygame.Surface:
        """"""
        return self._display

    @property
    def clock(self) -> pygame.time.Clock:
        """"""
        return self._clock

    @property
    def snake(self) -> List[Point]:
        """"""
        return self._snake

    @property
    def head(self) -> Point:
        """"""
        return self._head

    @head.setter
    def head(self, value):
        """"""
        self._head = value

    @property
    def direction(self) -> Direction:
        """"""
        return self._direction

    @direction.setter
    def direction(self, value):
        """"""
        self._direction = value

    @property
    def food(self) -> Point:
        """"""
        return self._food
    
    @food.setter
    def food(self, value):
        """"""
        self._food = value