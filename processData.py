import numpy as np

def calcHello():
    sum = 1 + 1
    return sum

def calcAdeus(arg):
    result = float(np.power(int(arg), 2))
    return result

def calcEndpoint(arg):
    result = []
    for num in arg:
        result.append(float(np.power(num, 2)))
    return result