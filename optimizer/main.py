import argparse
import os
import random
import sys
import subprocess
import json

import numpy as np 

from optimizer.algorithms import get_algorithm
from optimizer.deployment import deploy_kangaroo, deploy_single
from optimizer.evaluator import Simulator
from optimizer.logger import Logger

from mpi4py import MPI

def make_deterministic(seed): 
    '''Makes that each process has a different real seed'''
    Me = comm.Get_rank()
    real_seed = seed*(Me + 1)
    random.seed(real_seed)
    np.random.seed(real_seed)
    print(f'\nreal seed: {real_seed}\n')    
        
def run_algorithm(algorithm, args, comm, evaluation_session):
    Me = comm.Get_rank()
    best_solution, best_cost, path = algorithm.run(args.steps, evaluation_session)


    if best_cost is not None:
        print('\n\nPath taken:')
        for sol in path:
            print(sol[1], end=' ')
            sol[0].display()

        print('\n\nBest solution found:', end=' ')
        best_solution.display()
        print(best_cost)

        TabE = comm.gather(best_cost,root=0)
        TabS = comm.gather(best_solution,root=0)
        total_runs = comm.reduce(evaluation_session.run_counter,op=MPI.SUM, root=0)
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
            print('Total cost evaluations:', total_runs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimizer Launcher')
    parser.add_argument('--algorithm', type=str, default='hill_climbing')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('--kangaroo', action='store_true',
                        help='Run in parallel with different initializations')
    parser.add_argument('--hparams', type=str, default='{}',
                        help='JSON-serialized hparams dict')
    parser.add_argument('--problem_size', type=int, nargs=3, default=[256, 256, 256], help='Problem size')

    # usually you dont need to change this
    parser.add_argument('--phase', type=str,
                        default='deploy', choices=['deploy', 'run'])

    args = parser.parse_args()
    hparams = json.loads(args.hparams)

    comm = MPI.COMM_WORLD
    logger = Logger(process_id=comm.Get_rank(), logfile="mytest.log")

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
    
    print('Hyperparameters:')
    for k, v in sorted(algorithm.hparams.items()):
        print('\t{}: {}'.format(k, v))

    algorithm_class = get_algorithm(args.algorithm)
    algorithm = algorithm_class(hparams, args.problem_size, comm, logger)

    make_deterministic(args.seed)

    if args.phase == 'deploy' and args.kangaroo:
        deploy_kangaroo(args, sys.argv[0])
    elif args.phase == 'deploy' and not args.kangaroo:
        deploy_single(args, sys.argv[0])
    else: # phase is run
        evaluation_session = Simulator()
        run_algorithm(algorithm, args, comm, evaluation_session)
