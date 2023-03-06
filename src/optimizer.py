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
    logger.write_info(f'real seed: {real_seed}')    
        
def run_algorithm(args):
    algorithm_class = get_algorithm(args.algorithm)
    algorithm = algorithm_class(hparams, args.problem_size, logger)
    best_solution, best_cost, path = algorithm.run(args.steps)
    
    logger.write_info('Path taken:')
    for sol in path:
        logger.write_raw('\t' + str(sol[1]) + ' ' + sol[0].get_compilation_flags())

    logger.write_info('Best solution found:')
    logger.write_raw('\t' + str(best_cost) + ' ' + best_solution.get_compilation_flags())

    TabE = comm.gather(best_cost,root=0)
    TabS = comm.gather(best_solution,root=0)
    if (Me == 0):
        logger.jumpline()
        logger.write_info('Gathering solutions from all processes')
        logger.write_info('Best solutions:')
        for i in range(len(TabE)):
            logger.write_raw('\t' + str(TabE[i]) + ' ' + TabS[i].get_compilation_flags())

        logger.write_info('Best overall:')
        Eopt = max(TabE)
        idx = TabE.index(Eopt)
        Sopt = TabS[idx]
        logger.write_raw('\t' + str(Eopt) + ' ' + Sopt.get_compilation_flags())

if __name__ == "__main__":
    logger = Logger(process_id=comm.Get_rank(), logfile="myLog.log")

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

    logger.write_info('Args:')
    for k, v in sorted(vars(args).items()):
        logger.write_raw('\t{}: {}'.format(k, v))
    
    logger.write_info('Hyperparameters:')
    for k, v in sorted(hparams.items()):
        logger.write_raw('\t{}: {}'.format(k, v))


    make_deterministic(args.seed)

    if args.phase == 'deploy' and args.kangaroo:
        deploy_kangaroo(args, sys.argv[0], logger)
    elif args.phase == 'deploy' and not args.kangaroo:
        deploy_single(args, sys.argv[0], logger)
    else: # phase is run
        run_algorithm(args)