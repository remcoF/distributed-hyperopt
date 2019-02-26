import math
from time import sleep

def obj(params):
    x = params['x']
    sleep(3)
    return math.sin(x)