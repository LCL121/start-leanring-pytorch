# step1: pip install visdom
# step2: python -m visdom.server

from visdom import Visdom
import random
import math


def getY(x):
    return math.log(x)


viz = Visdom()
# Y, X, ID, {'title': 't'}
viz.line([math.log(0.01)], [0.01], win="try_line", opts=dict(titile='log'))
for x in range(1, 100):
    x = x * 0.01
    viz.line([getY(x)], [x], win='try_line', update='append')
for x in range(1, 10):
    viz.line([getY(x)], [x], win='try_line', update='append')


