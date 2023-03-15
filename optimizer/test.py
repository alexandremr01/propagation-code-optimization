from optimizer.solution import Solution

sol = Solution('-O3', 'avx', 256, 256, 256, 16, 256, 8, 8)

neigh = sol.get_neighbors()
for s in neigh:
    s.display()