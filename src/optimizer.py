import argparse
import os
import random

import numpy as np 
from algorithm_registry import get_algorithm
from mpi4py import MPI
comm = MPI.COMM_WORLD

def make_deterministic(seed): 
    '''Makes that each process has a different real seed'''
    Me = comm.Get_rank()
    real_seed = seed*(Me + 1)
    random.seed(real_seed)
    np.random.seed(real_seed)
    print(f'\n\treal seed: {real_seed}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimizer Launcher')
    parser.add_argument('--algorithm', type=str, default='hill_climbing')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=33)

    # usually you dont need to change this
    parser.add_argument('--phase', type=str, default='deploy', choices=['deploy', 'run'])

    args = parser.parse_args()

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    make_deterministic(args.seed)

    algorithm_class = get_algorithm(args.algorithm)
    algorithm = algorithm_class(args)
    algorithm.run(args.steps)
