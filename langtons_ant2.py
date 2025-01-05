# -*- mode:python;coding:utf-8 -*-
import argparse
from enum import IntEnum
import random
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Direction(IntEnum):
    N = 0
    E = 1
    W = 2
    S = 3
    def right_rotated(self) -> 'Direction':
        return (Direction.E if self == Direction.N else
                Direction.S if self == Direction.E else
                Direction.W if self == Direction.S else
                Direction.N)
    def left_rotated(self) -> 'Direction':
        return (Direction.W if self == Direction.N else
                Direction.N if self == Direction.E else
                Direction.E if self == Direction.S else
                Direction.S)
    def forward(self, y : int, x : int) -> tuple[int, int]:
        if self == Direction.N:
            y -= 1
        elif self == Direction.E:
            x += 1
        elif self == Direction.S:
            y += 1
        else:
            x -= 1
        return y, x

def generate_random_direction() -> 'Direction':
    return Direction(random.randint(0, 3))

class Ant:
    def __init__(self, y : int, x : int,
                 direction: 'Direction',
                 color1 : int,
                 color2 : int) -> None:
        self.y = y
        self.x = x
        self.direction = direction
        self.color1 = color1
        self.color2 = color2
    def move(self,
             field : np.ndarray[int, np.dtype[np.int16]],
             f_image : np.ndarray[int, np.dtype[np.int16]]) -> None:
        cell = field[self.y, self.x]
        f_image[self.y, self.x] = cell
        if cell == self.color1 or cell == 0:
            # At an initial square or a `color1` square,
            # turn 90° clockwise,
            # change the color of the square to `color2`.
            self.direction = self.direction.right_rotated()
            field[self.y, self.x] = self.color2
        if cell == self.color2:
            # At a `color2` square,
            # turn 90° counterclockwise,
            # change the color of the square to `color1`.
            self.direction = self.direction.left_rotated()
            field[self.y, self.x] = self.color1
        # Move forward one unit.
        y, x = self.direction.forward(self.y, self.x)
        self.y = y % np.shape(field)[0]
        self.x = x % np.shape(field)[1]
        # Display current position.
        f_image[self.y, self.x] = self.color2

HEIGHT = 400
WIDTH = 400

def update(count : int, max_count : int) -> list[matplotlib.artist.Artist]:
    sys.stdout.write('\r{} / {}'.format(count, max_count))
    for ant in ants:
        ant.move(field, f_image)
    plt.cla()
    img = plt.imshow(f_image, cmap = 'tab20b', interpolation = 'nearest')
    img.axes.get_xaxis().set_visible(False)
    img.axes.get_yaxis().set_visible(False)
    plt.text(10, HEIGHT - 20, "step: {} / {}".format(count, max_count))
    return [img]
def generate_animation(max_count : int, target : str) -> None:
    print('generating animation...')
    a = animation.FuncAnimation(fig,
                                update,
                                fargs = (max_count,),
                                interval = 1,
                                blit = True,
                                frames = max_count,
                                repeat = False)
    print('saving...')
    a.save(target, writer = "ffmpeg")

def simulate(max_count : int) -> None:
    for i in range(max_count):
        update(i, max_count)
        plt.pause(0.001)
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--animation",
                    type = str,
                    help = "Generate mp4 animation (ex. animation.mp4)")
parser.add_argument("-c", "--count",
                    type = int,
                    default = 30000,
                    help = "Max step count (default 30000)")
parser.add_argument("-s", "--seed",
                    type = int,
                    default = 8,
                    help = "Random seed")
args = parser.parse_args()
random.seed(args.seed)

field = np.zeros((HEIGHT, WIDTH), dtype = np.int16)
f_image = np.zeros((HEIGHT, WIDTH), dtype = np.int16)
ants = [Ant(random.randint(0, HEIGHT - 1),
            random.randint(0, WIDTH - 1),
            generate_random_direction(),
            n * 2,
            n * 2 + 1)
        for n in range(1,20)]
fig = plt.figure(figsize = (8, 8))
plt.axis('off')

if args.animation:
    generate_animation(args.count, args.animation)
else:
    simulate(args.count)
print('done')
