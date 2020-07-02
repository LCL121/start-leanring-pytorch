from visdom import Visdom
import math


def getY1(x):
    return math.log(x)


def getY2(x):
    return math.log2(x)


def getY3(x):
    return math.log10(x)


viz = Visdom()
viz.line([[math.log(0.01), math.log2(0.01), math.log10(0.01)]], [0.01],
         win="try_lines", opts=dict(titile='log & log2 & log10',
                                    legend=['log', 'log2', 'log10']))
for x in range(1, 100):
    x = x * 0.01
    viz.line([[getY1(x), getY2(x), getY3(x)]], [x], win='try_lines', update='append')
for x in range(1, 10):
    viz.line([[getY1(x), getY2(x), getY3(x)]], [x], win='try_lines', update='append')

