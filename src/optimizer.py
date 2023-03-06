import argparse
import os
import random
import sys
import subprocess
import json

import numpy as np 
from algorithm_registry import get_algorithm
from deployment import deploy_kangaroo, deploy_single

from logger import Logger

from mpi4py import MPI
comm = MPI.COMM_WORLD

def make_deterministic(seed): 
    '''Makes that each process has a different real seed'''
    Me = comm.Get_rank()
    real_seed = seed*(Me + 1)
    random.seed(real_seed)
    np.random.seed(real_seed)
    print(f'\nreal seed: {real_seed}\n')    
        
def run_algorithm(args):
    algorithm_class = get_algorithm(args.algorithm)
    algorithm = algorithm_class(hparams, args.problem_size, logger)
    best_solution, best_cost, path = algorithm.run(args.steps)
    
    print('\n\nPath taken:')
    for sol in path:
        print(sol[1], end=' ')
        sol[0].display()

    print('\n\nBest solution found:', end=' ')
    best_solution.display()
    print(best_cost)

    TabE = comm.gather(best_cost,root=0)
    TabS = comm.gather(best_solution,root=0)
    if (Me == 0):
        print('\n\nBest solutions:')
        for i in range(len(TabE)):
            TabS[i].display()
            print(TabE[i])
        print('\nBest overall:')
        Eopt = max(TabE)
        idx = TabE.index(Eopt)
        Sopt = TabS[idx]
        Sopt.display()
        print(Eopt)

if __name__ == "__main__":
    logger = Logger(process_id=comm.Get_rank(), logfile="mytest.log")

    parser = argparse.ArgumentParser(description='Optimizer Launcher')
    parser.add_argument('--algorithm', type=str, default='hill_climbing')
    parser.add_argument('--steps', type=int, default=10, help='Number of steps')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('--kangaroo', action='store_true', help='Run in parallel with different initializations')
    parser.add_argument('--hparams', type=str, default='{}', help='JSON-serialized hparams dict')

    # usually you dont need to change this
    parser.add_argument('--phase', type=str, default='deploy', choices=['deploy', 'run'])

    args = parser.parse_args()
    hparams = json.loads(args.hparams)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
    
    print('Hyperparameters:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))


    make_deterministic(args.seed)

    if args.phase == 'deploy' and args.kangaroo:
        deploy_kangaroo(args, sys.argv[0])
    elif args.phase == 'deploy' and not args.kangaroo:
        deploy_single(args, sys.argv[0])
    else: # phase is run
        run_algorithm(args)