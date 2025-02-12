"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(a: float, b: float) -> float:
    return a*b

def id(a: float) -> float:
    return a

def add(a: float, b: float) -> float:
    return a+b

def neg(a: float) -> float:
    return -a

def lt(a: float, b: float) -> float:
    if a<b:
        return 1.0
    return 0

def eq(a: float, b: float) -> float:
    if a == b:
        return 1.0
    return 0

def max(a: float, b: float) -> float:
    if a > b:
        return a
    return b

def is_close(a: float, b: float) -> bool:
    if (a >= b and a - b < 1e-2):
        return True
    if (a < b and b-a < 1e-2):
        return True
    return False

def sigmoid(a: float) -> float:
    if a >= 0:
        return 1.0 / (1.0 + exp(-a))
    return exp(a) / (1.0 + exp(a))

def relu(a: float) -> float:
    return max(0, a)

def log(a: float) -> float:
    return math.log(a)

def exp(a: float) -> float:
    return math.exp(a)

def inv(a: float) -> float:
    return 1/a

def log_back(a: float, b: float) -> float:
    return b/a

def exp_back(a: float, b: float) -> float:
    return math.exp(a) * b

def inv_back(a: float, b: float) -> float:
    return -b / (a*a)

def relu_back(a: float, b: float) -> float:
    if a <= 0:
        return 0
    return b

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists
def map(fn: Callable[[float], float], data: Iterable[float]) -> Iterable[float]:
    return [fn(a) for a in data]

def zipWith(fn: Callable[[float, float], float], data1: Iterable[float], data2: Iterable[float]) -> Iterable[float]:
    return [ fn(a,b) for (a,b) in zip(data1, data2) ]

def reduce(fn: Callable[[float, float], float], init: float, data: Iterable[float]) -> float:
    for a in data:
        init = fn(init, a)
    return init

# TODO: Implement for Task 0.3.
def negList(data: Iterable[float]) -> Iterable[float]:
    return map(neg, data)

def addLists(data1: Iterable[float], data2: Iterable[float]) -> Iterable[float]:
    return zipWith(add, data1, data2)

def sum(data: Iterable[float]) -> float:
    return reduce(add, 0, data)

def prod(data: Iterable[float]) -> float:
    return reduce(mul, 1, data)