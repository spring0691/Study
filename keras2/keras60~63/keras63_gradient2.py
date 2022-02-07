# import numpy as np

f = lambda x: x**2 - 4*x + 6

def f2(x):
    temp = x**2 - 4*x + 6
    return temp

gradient = lambda x: 2*x -4

def gradient2(x):
    temp = 2*x-4
    return temp

# 둘은 같다.

# 미분 -> 2차 3차함수에서 각 지점에 대한 기울기.

x = 0.0 # 초기값 