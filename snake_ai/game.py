#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the code for the Snake game.
"""


import os
import random
from enum import Enum
from typing import List

import pygame


# Initialize PyGame
# pygame.init()

pygame.display.init()
print('initialized display')
pygame.font.init()
print('initialized fonts')
# pygame.mixer.init()
# print('initialized mixer')
pygame.joystick.init()
print('initialized joystick')

# Get the path to the data directory
PARENT_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
DATA_DIRECTORY = os.path.join(PARENT_DIRECTORY, 'data')

# Get the path to import files in the data directory
ROBOTO_REGULAR_PATH = os.path.join(DATA_DIRECTORY, 'Roboto-Regular.ttf')

# Define the colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
LIGHT_GREEN = (0, 180, 0)
DARK_GREEN = (0, 130, 0)

# Define the fonts
roboto = pygame.font.Font(ROBOTO_REGULAR_PATH, 25)


class Direction(Enum):
    """"""

    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Point:
    """"""

    def __init__(self, x: int, y: int):
        """"""

        # Store user inputs
        self._x = int(x)
        self._y = int(y)
    
    @property
    def x(self) -> int:
        """"""
        return self._x
    
    @x.setter
    def x(self, value: int):
        """"""
        self._x = int(value)
    
    @property
    def y(self) -> int:
        """"""
        return self._y
    
    @y.setter
    def y(self, value: int):
        """"""
        self._y = int(value)
    
    def __eq__(self, other: "Point"):
        """"""
        return (self.x == other.x) and (self.y == other.y)

    def __repr__(self):
        """"""
        return f"Point(x={self.x}, y={self.y})"
    
    def __str__(self):
        """"""
        return f"Point({self.x}, {self.y})"


class SnakeGame:
    """"""

    _GAME_TITLE = "Snake"
    _BLOCK_SIZE = 20
    _BLOCK_PADDING = 4
    _BASE_CLOCK_SPEED = 10
    _CLOCK_SPEED = _BASE_CLOCK_SPEED

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
            Point(self.head.x-(0*self._BLOCK_SIZE), self.head.y),
            Point(self.head.x-(1*self._BLOCK_SIZE), self.head.y),
            Point(self.head.x-(2*self._BLOCK_SIZE), self.head.y),
        ]

        self._score = 0
        self._food = None
        self._place_food()
    
    def _place_food(self):
        """"""
        x = random.randint(0, (self._width-self._BLOCK_SIZE)//self._BLOCK_SIZE) * self._BLOCK_SIZE
        y = random.randint(0, (self._height-self._BLOCK_SIZE)//self._BLOCK_SIZE) * self._BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _update_ui(self):
        """"""

        self.display.fill(BLACK)

        for point in self.snake:
            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(
                point.x, point.y, self._BLOCK_SIZE, self._BLOCK_SIZE
            ))
            inner_block_size = self._BLOCK_SIZE-2*self._BLOCK_PADDING
            pygame.draw.rect(self.display, LIGHT_GREEN, pygame.Rect(
                point.x+self._BLOCK_PADDING, point.y+self._BLOCK_PADDING,
                inner_block_size, inner_block_size
            ))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, self._BLOCK_SIZE, self._BLOCK_SIZE
        ))
        
        text = roboto.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [5, 5])
        
        text = roboto.render(f"Speed: {self._CLOCK_SPEED}", True, WHITE)
        self.display.blit(text, [5, 35])

        pygame.display.flip()

    def _move_snake(self, direction: Direction):
        """"""
        
        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += self._BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= self._BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += self._BLOCK_SIZE
        elif direction == Direction.UP:
            y -= self._BLOCK_SIZE
        
        self.head = Point(x, y)

    def _check_for_collision(self):
        """"""
        # Check if snake hits the boundary
        if (self.head.x > (self._width - self._BLOCK_SIZE)) or (self.head.x < 0):
            return True
        if (self.head.y > (self._height - self._BLOCK_SIZE)) or (self.head.y < 0):
            return True

        # Check if snake hits itself
        if self.head in self.snake[1:]:
            return True
        
        # Otherwise, there was no collision
        return False

    def _update_clock_speed(self):
        """"""
        
        # 
        additional = self.score // 3
        self._CLOCK_SPEED = self._BASE_CLOCK_SPEED + additional

    def step(self):
        """"""

        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    if self.direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    if self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    if self.direction != Direction.UP:
                        self.direction = Direction.DOWN
                break

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
        self._update_clock_speed()
        self.clock.tick(self._CLOCK_SPEED)

        # Return game over and score
        return game_over, self.score

    def play(self, print_results=True):
        """"""

        # Start the game loop
        while True:
            game_over, score = self.step()
            if game_over:
                break
        
        # Print the final results if desired
        if print_results:
            print(f"Final score: {score}")
            print(f"Final speed: {self._CLOCK_SPEED}")
        
        # Quit the game
        self.quit()
    
    def quit(self):
        """"""

        pygame.quit()

    @property
    def score(self) -> int:
        """"""
        return self._score
    
    @score.setter
    def score(self, value: int):
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
    def head(self, value: Point):
        """"""
        self._head = value

    @property
    def direction(self) -> Direction:
        """"""
        return self._direction

    @direction.setter
    def direction(self, value: Direction):
        """"""
        self._direction = value

    @property
    def food(self) -> Point:
        """"""
        return self._food
    
    @food.setter
    def food(self, value: Point):
        """"""
        self._food = value