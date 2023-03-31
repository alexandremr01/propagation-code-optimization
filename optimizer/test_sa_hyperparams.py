import nevergrad as ng

import argparse
import os
import random
import sys
import subprocess
import json
import time

import numpy as np

from optimizer import evaluators
from optimizer.algorithms import get_algorithm, ALGORITHMS
from optimizer.deployment import deploy_kangaroo, deploy_single
from optimizer.logger import Logger, find_slurmfile, slurm_to_logfile

from mpi4py import MPI


def evaluate_hparams(hparams, args, comm, logger):

    # ad hoc
    hparams['lambda'] = hparams.pop('lbd')

    algorithm_class = get_algorithm(args.algorithm)
    algorithm = algorithm_class(hparams, args.problem_size, comm, logger, args.flexible_shape)

    print('Hyperparameters:')
    for k, v in sorted(algorithm.hparams.items()):
        print('\t{}: {}'.format(k, v))

    make_deterministic(args.seed)

    evaluation_session = evaluators.Simulator(logger)
    evaluator = evaluators.NaiveEvaluator(args.program_path, evaluation_session)
    best_solution, best_cost, path = algorithm.run(args.steps, evaluator)
    print('Best solution found:')
    print('\t' + str(best_cost) + ' ' + best_solution.get_compilation_flags())

    return -best_cost


def make_deterministic(seed):
    '''Makes that each process has a different real seed'''
    Me = comm.Get_rank()
    real_seed = seed * (Me + 1)
    random.seed(real_seed)
    np.random.seed(real_seed)
    logger.write_info(f'real seed: {real_seed}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimizer Launcher')
    parser.add_argument('--algorithm', type=str, choices=ALGORITHMS.keys(), default='simulated_annealing')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed') # Use a suboptimal seed
    parser.add_argument('--batch', action='store_true',
                        help='Uses multiple nodes')
    parser.add_argument('--problem_size', type=int, nargs=3, default=[256, 256, 256],
                        help='Three dimensions of problem size')
    parser.add_argument('--flexible_shape', action='store_true', help='Allows changing the problem shape')
    parser.add_argument('--use_energy', action='store_true', help='Use energy consumption in the cost function')
    parser.add_argument('--phase', type=str, default='deploy', choices=['deploy', 'run'])
    parser.add_argument('--program_path', type=str, default='iso3dfd-st7', help='Folder that contains program code')
    parser.add_argument('--log', type=str, default='myLog.log', help='Name of the log file (with extension)')

    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    logger = Logger(process_id=comm.Get_rank(), save_to_logfile=False, save_to_terminal=False)

    instrum = ng.p.Instrumentation(t0=ng.p.Scalar(lower=1, upper=1000), lbd=ng.p.Scalar(lower=0.8, upper=1))
    optimizer = ng.optimizers.BO(parametrization=instrum, budget=20, num_workers=1)

    for _ in range(optimizer.budget):
        x = optimizer.ask()
        loss = evaluate_hparams(x.kwargs, args, comm, logger)
        optimizer.tell(x, loss)

    recommendation = optimizer.provide_recommendation()
    print("Recommendation:")
    print(recommendation.value)
