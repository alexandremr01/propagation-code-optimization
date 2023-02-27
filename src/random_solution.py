import numpy as np

from solution import Solution

def get_random_solution(problem_size_x, problem_size_y, problem_size_z):
    o_levels = ['-O2', '-O3', '-Ofast']
    simds = ['avx', 'avx2', 'avx512']
    n_threads = [str(x) for x in [8, 16, 32]]
    threadblocks = [str(x) for x in [2, 4, 8, 16, 32, 64, 128, 256, 512]]
    return Solution(
        olevel=np.random.choice(o_levels),
        simd=np.random.choice(simds),
        problem_size_x=problem_size_x,
        problem_size_y=problem_size_y,
        problem_size_z=problem_size_z,
        nthreads=np.random.choice(n_threads),
        thrdblock_x=np.random.choice(threadblocks[3:]), # must be multiple of 16
        thrdblock_y=np.random.choice(threadblocks),
        thrdblock_z=np.random.choice(threadblocks),
    )

