import numpy as np

def sum(a, b, c, d):
    print(a+b+c+d)

c = {"a": 1,"b": 2,"c": 3,"d": 4}
sum(**c)