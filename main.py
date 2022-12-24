import argparse

import pygame

from snake_ai import agent, game


if __name__ == '__main__':

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Optional argument to play manually
    parser.add_argument('-m', '--manual', action='store_true', dest='manual')

    # Read arguments from command line
    args = parser.parse_args()

    # Play the game
    if args.manual:
        game.SnakeGame().play()
    else:
        agent.train()