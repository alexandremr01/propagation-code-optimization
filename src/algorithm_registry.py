from hill_climbing import HillClimbing
from successive_descents import SuccessiveDescents

ALGORITHMS = {
    'hill_climbing': HillClimbing,
    'successive_descents': SuccessiveDescents,
}

def get_algorithm(algorithm_name):
    if ALGORITHMS.get(algorithm_name) is None:
        raise NotImplementedError
    return ALGORITHMS[algorithm_name]