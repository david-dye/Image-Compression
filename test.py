import numpy as np
import matplotlib.pyplot as plt

def main():
    key = 0
    b = bin(key)
    print(b)
    if key < 0:
        b = "0" * (len(b)-2) + "11" + b[3:]
    else:
        b = "0" * (len(b)-1) + "10" + b[2:]
    print(b)



def get_lambdas(x):
    return (1 - 10**(-x))




if __name__ == "__main__":
    main()