import numpy as np

def example(x):
    return np.sum(x**2)


def example_grad(x):
    return 2*x

def foo(x):
    result = 1
    λ = 4 # this is here to make sure you're using Python 3
    for x_i in x:
    	result += x_i**λ
    return result


def foo_grad(x):
    for x_i in x:
    	idx = np.argwhere(x == x_i)
    	x[idx] = 4*x_i**3
    return x

def bar(x):
    return np.prod(x)

def bar_grad(x):
	res = np.prod(x)
	for x_i in x:
		idx = np.argwhere(x == x_i)
		x[idx] = res / x_i
	return x