import numpy as np

def func1(N):
    for i in range(N):
        print("Hello!")

def func2(N):
    x = np.zeros(N)
    x += 1000
    return x

def func3(N):
    x = np.zeros(1000)
    x = x * N
    return x

def func4(N):
    x = 0
    for i in range(N):
        for j in range(i,N):
            x += (i*j)
    return x