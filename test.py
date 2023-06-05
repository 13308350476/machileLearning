from typing import Any
import torch
import math

def reverse(input):
    compose = input.split(' ')
    compose = compose[-1::-1]
    print(type(compose))
    output = ' '.join(compose)
    print(type(output))
    return output

class Foobar:
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds):
        print(str(args[0]))





if __name__ == "__main__":
    '''
    input = "i like py"
    rw = reverse(input)
    print(rw)
    '''
    a = 1
    tup = (a,2,3,4)
    a = 5
    print(type(tup))

    x = torch.linspace(-math.pi, math.pi, 4)
    print(x.shape)
    print(x)
    p = torch.tensor([1, 2, 3])
    print(p.shape)
    print(p)
    xx = x.unsqueeze(-1).pow(p)
    print(xx.shape)
    print(xx)

    print(0.2 % 1000)