from typing import Any
# import torch
import math
import gzip
import csv

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
    file_name = './datasets/name/names_train.csv.gz'
    with gzip.open(file_name, 'rt') as f:
        reader = csv.reader(f)
        rows = list(reader)
    print(rows)