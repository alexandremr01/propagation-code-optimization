import cma
from solution import Solution
# function to optimize

max_threads = 32
max_thrdblock_x = 256
max_thrdblock_y = 256
max_thrdblock_z = 256
olevels = ['-O2', '-O3', '-Ofast']
simds = ['avx', 'avx2', 'avx512']
boudary = {'bounds': [0, [2.99, 2.99, max_threads//8,
                          max_thrdblock_x//8, max_thrdblock_y//8, max_thrdblock_z//8]]}


param = {'olevels': '-O2', 'simds': 'avx', 'problem_size_x': 256, 'problem_size_y': 256,
         'problem_size_z': 256, 'n_threads': 16, 'thrdblock_x': 16, 'thrdblock_y': 16, 'thrdblock_z': 16}


def x_to_param(x):
    param['olevels'] = olevels[int(x[0])]
    param['simds'] = simds[int(x[1])]
    param['n_threads'] = 8*int(x[2]+1)
    param['thrdblock_x'] = 8*int(x[3]+1)
    param['thrdblock_y'] = 8*int(x[4]+1)
    param['thrdblock_z'] = 8*int(x[5]+1)
    return param


def cost_function(x):
    param = x_to_param(x)
    print(f'Computing of the cost of : {param}')
    sol = Solution(param['olevels'], param['simds'], str(param['problem_size_x']), str(param['problem_size_y']),
                   str(param['problem_size_z']), str(param['n_threads']), param['thrdblock_x'], param['thrdblock_y'], param['thrdblock_z'])

    return -sol.cost()


if __name__ == "__main__":
    x0 = 6 * [0]  # initial solution
    sigma0 = 1   # initial standard deviation to sample new solutions
    xopt, es = cma.fmin2(cost_function, x0, sigma0, {'bounds': [0, [
        2.99, 2.99, max_threads//8, max_thrdblock_x//8, max_thrdblock_y//8, max_thrdblock_z//8]]})
