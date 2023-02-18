
from utils import *

if __name__=='__main__':
    args = parse()

    from data.colored_mnist import colored_mnist_gen
    colored_mnist_gen(args)
    
    
