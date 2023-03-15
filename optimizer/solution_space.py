class SolutionSpace:
    o_levels = ['-O2', '-O3', '-Ofast']
    simds = ['avx', 'avx2', 'avx512']
    nthreads = [16]
    problem_size = [32, 64, 128, 256, 512, 1024, 2048]
    # Reasoning for threadblock sizes:
    # L1 cache is 512KiB, L2 cache is 16MiB, L3 cache is 22 MiB
    # size of cache: product of 3 dimensions * 4 bytes/integer * 3 arrays
    # 512 * 16 * 16 * 4 * 3  = 1.5 MiB
    # 2048 * 16 * 16 * 4 * 3 = 6 MiB
    # with sizes problem_size * 16 * 16, it always fit in the L3 at least
    threadblocks = list(range(1, 17))

