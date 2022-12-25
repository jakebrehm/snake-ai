# Snake AI

A custom version of the classic snake game, played either manually or by a neural network AI.

## Requirements

The following are the main libraries that are required before running the project:

- [pygame](https://github.com/pygame/pygame)
- [PyTorch](https://github.com/pytorch/pytorch)
- [matplotlib](https://github.com/matplotlib/matplotlib)

## Getting started

To being, either download the repository as a zip file or clone the project using
```
git clone https://github.com/jakebrehm/snake-ai.git
```

Then, run the program using the following commands:
```
cd snake-ai
pip3 install -r requirements.txt
python3 main.py
```

## About the game

The game is a custom implementation of the classic Snake game, and it was created using [pygame](https://github.com/pygame/pygame).

To play the game manually, use the `-m` or `--manual` command line arguments:
```
python3 main.py -m
```

## About the model

The model is a neural network composed of three layers (1 hidden layer). The input layer has 11 nodes, the hidden layer has a changeable 256 nodes, and the output layer has 3 nodes.

The 11 boolean inputs are as follows:
- Danger straight ahead
- Danger to the right
- Danger to the left
- Snake is moving to the left
- Snake is moving to the right
- Snake is moving up
- Snake is moving down
- There is food to the left
- There is food to the right
- There is food above
- There is food below

The 3 boolean outputs are as follows:
- Go straight ahead
- Turn right
- Turn left

These outputs are used to determined the next move of the snake.

## Training results

<p align="center">
  <img src="https://raw.githubusercontent.com/jakebrehm/snake-ai/master/img/demo.gif"
  alt="Snake AI Demo"/>
</p>

## Future improvements

- Save the results to a csv file for post hoc analysis and/or visualization.
- The snake will not become better after a certain point because it nevers learns how to not trap itself in its own body. Figure out a way to avoid this.
- Allow for the ability to press a button to kill the snake/stop the training process, so that the user doesn't have to manually close the window.

## Authors

- **Jake Brehm** - [Email](mailto:mail@jakebrehm.com) | [Github](http://github.com/jakebrehm) | [LinkedIn](http://linkedin.com/in/jacobbrehm)