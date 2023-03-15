class SolutionSpace:
    o_levels = ['-O2', '-O3', '-Ofast']
    simds = ['avx', 'avx2', 'avx512']
    nthreads = [16]
    # L1 cache is 512KiB (512*1024 bytes = 524288 bytes).
    # worst case: first dimension * 16 * 16 * 4 = 524288 < 512
    # First dimension 512: 1024 bytes for cache
    # First dimension 1024: 512 bytes for cache
    # First dimension 2048: 256 bytes for cache
    # all of
    threadblocks = list(range(1, 17))

    problem_size = [32, 64, 128, 256, 512, 1024, 2048]