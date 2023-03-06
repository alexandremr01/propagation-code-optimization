from hill_climbing import HillClimbing
from greedy import Greedy, TabuGreedy
from local_conditionnal_acceptance import LocalConditionnalAcceptance

ALGORITHMS = {
    'hill_climbing': HillClimbing,
    'greedy': Greedy,
    'tabu_greedy': TabuGreedy,
    'simulated_annealing': LocalConditionnalAcceptance,
}

def get_algorithm(algorithm_name):
    if ALGORITHMS.get(algorithm_name) is None:
        raise NotImplementedError
    return ALGORITHMS[algorithm_name]