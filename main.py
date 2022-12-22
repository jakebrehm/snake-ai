import pygame

from snake_ai import agent, game


if __name__ == '__main__':

    # User-specified constants
    MANUAL = False

    # Play the game
    if MANUAL:
        game.SnakeGame().play()
    else:
        agent.train()