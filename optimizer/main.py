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
        logger.write_info('Path taken:')
        for sol in path:
            logger.write_raw('\t' + str(sol[1]) + ' ' + sol[0].get_compilation_flags())

        logger.write_info('Best solution found:')
        logger.write_raw('\t' + str(best_cost) + ' ' + best_solution.get_compilation_flags())

        TabE = comm.gather(best_cost,root=0)
        TabS = comm.gather(best_solution,root=0)
        total_runs = comm.reduce(evaluation_session.run_counter,op=MPI.SUM, root=0)
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
            logger.write_info(f'Total cost evaluations: {total_runs}')

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

    logger.write_info('Args:')
    for k, v in sorted(vars(args).items()):
        logger.write_raw('\t{}: {}'.format(k, v))
    
    logger.write_info('Hyperparameters:')
    for k, v in sorted(hparams.items()):
        logger.write_raw('\t{}: {}'.format(k, v))

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
