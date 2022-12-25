#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper module for plotting the results of the snake game.
"""


import matplotlib.pyplot as plt
from IPython import display


def enable():
    """Enable interactive mode"""
    plt.ion()


def add(scores, mean_scores):
    """Plots the scores."""

    # Clear the current display
    display.clear_output(wait=True)

    # Display the current figure
    figure = plt.gcf()
    display.display(figure)

    # Clear the current figure
    plt.clf()

    # Use alternative theme
    plt.style.use('seaborn-v0_8-darkgrid')

    # Add labels to the plot
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plot the scores
    plt.scatter(list(range(1, len(scores)+1)), scores, c="blue")
    plt.plot(mean_scores, c="red")

    # Set axis limits
    plt.ylim(ymin=-0.05)

    # Add text to the plot showing the latest values
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))