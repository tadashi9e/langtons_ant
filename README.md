# Langton's Ant

Langton's Ant is a two-dimensional Turing machine with a very simple set of rules but complex emergent behavior. It was invented by Chris Langton in 1986 and runs on a square lattice of black and white cells. The concept has been generalized in several ways, such as turmites, which incorporate more colors and additional states ([Wikipedia](https://en.wikipedia.org/wiki/Langton%27s_ant)).

## Requirements

- Python3
- numpy
- matplotlib

## langtons_ant.py

- At a white square, turn 90° clockwise, flip the color of the square, and move forward one unit.
- At a black square, turn 90° counterclockwise, flip the color of the square, and move forward one unit.

## langtons_ant2.py

This version is almost the same as Langton's Ant, but each ant has its own individual `color1` (representing white) and `color2` (representing black).

- At an initial square or a `color1` square, turn 90° clockwise, change the color of the square to `color2`, and move forward one unit.
- At a `color2` square, turn 90° counterclockwise, change the color of the square to `color1`, and move forward one unit.
- At a square with any other color, move forward one unit (without changing the color of the square).
