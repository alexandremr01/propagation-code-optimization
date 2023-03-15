class SolutionSpace:
    o_levels = ['-O2', '-O3', '-Ofast']
    simds = ['avx', 'avx2', 'avx512']
    nthreads = [16]
    threadblocks = list(range(1, 17))

    problem_size = [32, 64, 128, 256, 512, 1024, 2048]