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

    # Add labels to the plot
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plot the scores
    plt.plot(scores)
    plt.plot(mean_scores)

    # Set axis limits
    plt.ylim(ymin=0)

    # Add text to the plot showing the latest values
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))