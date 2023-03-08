class SolutionSpace:
    o_levels = ['-O2', '-O3', '-Ofast']
    simds = ['avx', 'avx2', 'avx512']
    n_threads = [8, 16, 32]
    threadblocks = [2, 4, 8, 16, 32, 64, 128, 256, 512]