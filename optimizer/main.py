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

def make_deterministic(seed):
    '''Makes that each process has a different real seed'''
    Me = comm.Get_rank()
    real_seed = seed*(Me + 1)
    random.seed(real_seed)
    np.random.seed(real_seed)
    logger.write_info(f'real seed: {real_seed}')

def run_algorithm(algorithm, args, comm, evaluator):
    Me = comm.Get_rank()
    best_solution, best_cost, path = algorithm.run(args.steps, evaluator)
    TabE = comm.gather(best_cost,root=0)
    TabS = comm.gather(best_solution,root=0)
    start_time = time.time()
    total_runs = comm.reduce(evaluator.get_counter(),op=MPI.SUM, root=0)
    if best_cost is not None:
        logger.write_info('Path taken:')
        for sol in path:
            logger.write_raw('\t' + str(sol[1]) + ' ' + sol[0].get_compilation_flags())

        logger.write_info('Best solution found:')
        logger.write_raw('\t' + str(best_cost) + ' ' + best_solution.get_compilation_flags())

        if (Me == 0):
            comm.Barrier() # guarantee that these will be the final messages
            TabE = [x for x in TabE if x is not None]
            TabS = [x for x in TabS if x is not None]
            logger.jumpline()
            logger.write_info('Gathering solutions from all processes')
            logger.write_info('Best solutions:')

            best_ix = 0
            best_E_overall = -1
            for i in range(len(TabE)):
                recalculated_cost = evaluator.cost(TabS[i], num_evaluations=3, ignore_cache=True)
                logger.write_raw('\t' + str(TabE[i]) + ' ' + TabS[i].get_compilation_flags() + ' Final evaluation: ' + str(recalculated_cost))
                if recalculated_cost > best_E_overall:
                    best_E_overall = recalculated_cost
                    best_ix = i

            logger.write_info('Best overall:')
            logger.write_raw('\t' + str(best_E_overall) + ' ' + TabS[best_ix].get_compilation_flags())
            logger.write_info(f'Total cost evaluations: {total_runs}')
            final_time = time.time()
            elapsed_time = np.round(final_time - start_time, 2)
            logger.write_info(f'Elapsed time: {elapsed_time} seconds')
            return
    if (Me != 0):
        comm.Barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimizer Launcher')
    parser.add_argument('--algorithm', type=str, choices=ALGORITHMS.keys(), default='hill_climbing')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('--batch', action='store_true',
                        help='Uses multiple nodes')
    parser.add_argument('--hparams', type=str, default='{}',
                        help='JSON-serialized hyperparameters dictionary')
    parser.add_argument('--problem_size', type=int, nargs=3, default=[256, 256, 256], help='Three dimensions of problem size')
    parser.add_argument('--flexible_shape', action='store_true', help='Allows changing the problem shape')
    parser.add_argument('--use_energy', action='store_true', help='Use energy consumption in the cost function')
    parser.add_argument('--phase', type=str, default='deploy', choices=['deploy', 'run'])
    parser.add_argument('--program_path', type=str, default='iso3dfd-st7', help='Folder that contains program code')
    parser.add_argument('--log', type=str, default='myLog.log', help='Name of the log file (with extension)')

    args = parser.parse_args()
    hparams = json.loads(args.hparams)

    comm = MPI.COMM_WORLD

    logfile = args.log
    if args.batch:
        logger = Logger(process_id=comm.Get_rank(), save_to_logfile=False)
    else:
        logger = Logger(process_id=comm.Get_rank(), logfile=logfile)

    logger.write_info('Args:')
    for k, v in sorted(vars(args).items()):
        logger.write_raw('\t{}: {}'.format(k, v))

    algorithm_class = get_algorithm(args.algorithm)
    algorithm = algorithm_class(hparams, args.problem_size, comm, logger, args.flexible_shape)

    logger.write_info('Hyperparameters:')
    for k, v in sorted(algorithm.hparams.items()):
        logger.write_raw('\t{}: {}'.format(k, v))

    make_deterministic(args.seed)


    if args.phase == 'deploy' and args.batch:
        deploy_kangaroo(args, sys.argv[0], logger)
    elif args.phase == 'deploy' and not args.batch:
        deploy_single(args, sys.argv[0], logger)
    else: # phase is run
        evaluation_session = evaluators.Simulator(logger)
        evaluator = evaluators.NaiveEvaluator(args.program_path, evaluation_session)
        if args.use_energy:
            evaluator = evaluators.EnergyEvaluator(args.program_path, evaluation_session)
        run_algorithm(algorithm, args, comm, evaluator)
        logger.write_info(f"Run finished. Logs can be found at {logfile}.")
    
    # retrieve logs from slurm file
    if args.batch:
        slurm_to_logfile(find_slurmfile(os.getcwd()), logfile)