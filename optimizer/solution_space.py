class SolutionSpace:
    o_levels = ['-O2', '-O3', '-Ofast']
    simds = ['avx', 'avx2', 'avx512']
    nthreads = [16]
    threadblocks = list(range(1, 17))