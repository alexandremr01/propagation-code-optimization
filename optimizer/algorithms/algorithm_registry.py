from optimizer.algorithms.hill_climbing import HillClimbing
from optimizer.algorithms.greedy import Greedy, TabuGreedy
from optimizer.algorithms.local_conditionnal_acceptance import LocalConditionnalAcceptance
from optimizer.algorithms.curious_simulated_annealing import CuriousSimulatedAnnealing
from optimizer.algorithms.cmaes import CMAESAlgorithm
from optimizer.algorithms.ego import EfficientGlobalOptimization

ALGORITHMS = {
    'hill_climbing': HillClimbing,
    'greedy': Greedy,
    'tabu_greedy': TabuGreedy,
    'simulated_annealing': LocalConditionnalAcceptance,
    'csa': CuriousSimulatedAnnealing,
    'cmaes': CMAESAlgorithm,
    'ego': EfficientGlobalOptimization
}


def get_algorithm(algorithm_name):
    if ALGORITHMS.get(algorithm_name) is None:
        raise NotImplementedError
    return ALGORITHMS[algorithm_name]
