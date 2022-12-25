#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the code for the Snake game.
"""


import os
import random
from enum import Enum
from typing import List, Tuple

import numpy as np
import pygame
from dataclasses import dataclass


# Initialize PyGame
pygame.init()

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
roboto = pygame.font.Font(ROBOTO_REGULAR_PATH, 20)


class Direction(Enum):
    """Stores the direction of the snake."""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Action(Enum):
    """Representation of possible actions for the snake to take."""
    GO_STRAIGHT = [1, 0, 0]
    GO_RIGHT = [0, 1, 0]
    GO_LEFT = [0, 0, 1]


@dataclass
class Point:
    """Store coordinates for a point on the game screen."""
    x: int
    y: int


class SnakeGame:
    """Plays the classic snake game."""

    _GAME_TITLE = "Snake"
    _BLOCK_SIZE = 20
    _BLOCK_PADDING = 4
    _BASE_CLOCK_SPEED = 10
    _CLOCK_SPEED = _BASE_CLOCK_SPEED

    def __init__(self, width: int=640, height: int=480, start_length: int=3):
        """Initializes the SnakeGame instance."""
        
        # Store user inputs
        self._width = width
        self._height = height
        self._start_length = start_length

        # Initialize display
        self._display = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption(self._GAME_TITLE)
        self._clock = pygame.time.Clock()

        # Initialize game state
        self._initialize_game_state()

    def _initialize_game_state(self):
        """Set up the game."""

        # The snake should be moving right when the game starts
        self._direction = Direction.RIGHT
        # Start the snake in the middle of the game
        self._head = Point(self._width/2, self._height/2)
        # Start the snake with the specified length
        self.snake = self._generate_snake(self._start_length)

        # Reset the score and place food
        self._score = 0
        self._food = None
        self._place_food()

    def _generate_snake(self, length: int) -> List[Point]:
        """Generate the initial snake with variable body length."""

        x, y = self.head.x, self.head.y
        return [Point(x-(i*self._BLOCK_SIZE), y) for i in range(length)]

    def _place_food(self):
        """Places the food in a new, appropriate location."""
        
        # Get the size of each game block
        block = self._BLOCK_SIZE

        # Randomly determine a new location for the food
        x = random.randint(0, (self._width-block)//block) * block
        y = random.randint(0, (self._height-block)//block) * block
        self.food = Point(x, y)
        
        # Make sure the new food isn't inside the snake's body
        if self.food in self.snake:
            self._place_food()

    def _update_ui(self):
        """Updates the game's UI."""

        # Draw the black background first
        self.display.fill(BLACK)

        # Draw each part of the snake on the screen
        for point in self.snake:
            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(
                point.x, point.y, self._BLOCK_SIZE, self._BLOCK_SIZE
            ))
            inner_block_size = self._BLOCK_SIZE-2*self._BLOCK_PADDING
            pygame.draw.rect(self.display, LIGHT_GREEN, pygame.Rect(
                point.x+self._BLOCK_PADDING, point.y+self._BLOCK_PADDING,
                inner_block_size, inner_block_size
            ))
        
        # Draw the food on the screen
        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, self._BLOCK_SIZE, self._BLOCK_SIZE
        ))
        
        # Add the current score to the upper left corner
        text = roboto.render(f"Score: {self.score}", True, WHITE)
        text.set_alpha(100)
        self.display.blit(text, [5, 5])
        # Add the current speed to the upper left corner
        text = roboto.render(f"Speed: {self._CLOCK_SPEED}", True, WHITE)
        text.set_alpha(100)
        self.display.blit(text, [5, 30])

        # Update the entire display
        pygame.display.flip()

    def _move_snake(self, direction: Direction):
        """Move the head of the snake depending on the selected direction."""
        
        # Get current coordinates of snake's head
        x = self.head.x
        y = self.head.y

        # Modify coordinates depending on snake's direction
        if direction == Direction.RIGHT:
            x += self._BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= self._BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += self._BLOCK_SIZE
        elif direction == Direction.UP:
            y -= self._BLOCK_SIZE
        
        # Change the head coordinates to the new values
        self.head = Point(x, y)

    def check_for_collision(self):
        """Check if there are any collisions."""

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
        """Updates the current clock speed using an arbitrary algorithm."""
        
        # Arbitrarily divide the score by three, then add to the base speed
        additional = self.score // 3
        self._CLOCK_SPEED = self._BASE_CLOCK_SPEED + additional

    def step(self):
        """Play the next game step."""

        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
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
        self._move_snake(self.direction)
        # Update the head
        self.snake.insert(0, self.head)

        # Check if game over
        game_over = False
        if self.check_for_collision():
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

        # Return whether or not the game is over, and the score
        return game_over, self.score

    def play(self, print_results=True):
        """Start the game loop."""

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
        """Quits the game."""

        pygame.quit()

    @property
    def score(self) -> int:
        """Gets the score."""
        return self._score
    
    @score.setter
    def score(self, value: int):
        """Sets the score."""
        self._score = value

    @property
    def display(self) -> pygame.Surface:
        """Gets the display variable."""
        return self._display

    @property
    def clock(self) -> pygame.time.Clock:
        """Gets the clock variable."""
        return self._clock

    @property
    def snake(self) -> List[Point]:
        """Gets the snake list."""
        return self._snake

    @snake.setter
    def snake(self, value: List[Point]):
        """Sets the snake list."""
        self._snake = value

    @property
    def head(self) -> Point:
        """Gets the head."""
        return self._head

    @head.setter
    def head(self, value: Point):
        """Sets the head."""
        self._head = value

    @property
    def direction(self) -> Direction:
        """Gets the snake's direction."""
        return self._direction

    @direction.setter
    def direction(self, value: Direction):
        """Sets the snake's direction."""
        self._direction = value

    @property
    def food(self) -> Point:
        """Gets the food's position."""
        return self._food
    
    @food.setter
    def food(self, value: Point):
        """Sets the food's position."""
        self._food = value

    @property
    def block_size(self) -> int:
        """Gets the size of a game block."""
        return self._BLOCK_SIZE


class SnakeGameBot(SnakeGame):
    """"""

    def __init__(self,
        width: int=640,
        height: int=480,
        start_length: int=3,
        base_clock_speed: int=60,
    ):
        """Initializes the SnakeGameBot instance."""

        # Set clock speeds
        self._BASE_CLOCK_SPEED = base_clock_speed
        self._CLOCK_SPEED = self._BASE_CLOCK_SPEED

        # Initialize the parent class
        super().__init__(width=width, height=height, start_length=start_length)

        # Initialize frame number
        self._frame = 0
    
    def reset(self):
        """Reset to the game's original state."""
        self._initialize_game_state()
        self.frame = 0

    def _move_snake(self, action: Action):
        """Move the head of the snake depending on the selected direction."""
        
        # Define list of clockwise directions
        clockwise_directions = [
            Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP
        ]
        # Get index of current direction
        direction_index = clockwise_directions.index(self.direction)

        # Change direction if necessary
        if action == Action.GO_RIGHT:
            # Make a right turn --> clockwise change
            self.direction = clockwise_directions[(direction_index+1) % 4]
        elif action == Action.GO_LEFT:
            # Make a left turn --> counter-clockwise change
            self.direction = clockwise_directions[(direction_index-1) % 4]
        
        # Get current coordinates of snake's head
        x = self.head.x
        y = self.head.y

        # Modify coordinates depending on snake's direction
        if self.direction == Direction.RIGHT:
            x += self._BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= self._BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += self._BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= self._BLOCK_SIZE
        
        # Change the head coordinates to the new values
        self.head = Point(x, y)

    def check_for_collision(self, point: Point=None):
        """Wrapper around _check_for_collision().
        
        This is done so the original method can still be overwritten.
        """
        return self._check_for_collision(point)

    def _check_for_collision(self, point: Point=None):
        """Check if there are any collisions."""

        # Initialize the point variable
        if point is None:
            point = self.head

        # Check if snake hits the boundary
        if (point.x > (self._width - self._BLOCK_SIZE)) or (point.x < 0):
            return True
        if (point.y > (self._height - self._BLOCK_SIZE)) or (point.y < 0):
            return True

        # Check if snake hits itself
        if point in self.snake[1:]:
            return True
        
        # Otherwise, there was no collision
        return False

    def step(self, action: Action):
        """Play the next game step."""

        # Update frame number
        self.frame += 1

        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
                quit()

        # Move snake
        self._move_snake(action)
        # Update the head
        self.snake.insert(0, self.head)

        # Check if game over
        reward = 0
        game_over = False
        threshold = 100 * len(self.snake)
        if self.check_for_collision() or (self.frame > threshold):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Place new food or finalize move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # Update the pygame UI and the clock
        self._update_ui()
        self._update_clock_speed()
        self.clock.tick(self._CLOCK_SPEED)

        # Return the reward, whether or not the game is over, and the score
        return reward, game_over, self.score

    def play(self, print_results=True):
        """Start the game loop."""

        # Start the game loop
        while True:
            reward, game_over, score = self.step()
            if game_over:
                break
        
        # Print the final results if desired
        if print_results:
            print(f"Final score: {score}")
            print(f"Final speed: {self._CLOCK_SPEED}")
            print(f"Reward: {reward}")
        
        # Quit the game
        self.quit()

    @property
    def frame(self) -> int:
        """Gets the current frame."""
        return self._frame
    
    @frame.setter
    def frame(self, value: int):
        """Sets the current frame."""
        self._frame = value