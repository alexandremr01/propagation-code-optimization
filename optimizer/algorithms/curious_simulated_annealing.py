import random
import numpy as np
from optimizer.solution import Solution
from optimizer.random_solution import get_random_solution
from optimizer.algorithms import Algorithm

def acceptance_func(energy_diff, temp):
    return 1 / (1 - energy_diff / temp) # cost is good, so we need to invert the sign

def group_particles(particles, n_groups):
    new_particles = [[] for _ in range(n_groups)]
    for i, particle in enumerate(particles):
        new_particles[i%n_groups].append(particle)
    return new_particles

def ungroup_particles(particles):
    list_pointer = 0
    new_particles = []
    list_indexes = [0] * len(particles)
    while list_indexes[list_pointer] != len(particles[list_pointer]):
        index_in_list = list_indexes[list_pointer]
        new_particles.append(particles[list_pointer][index_in_list])
        list_indexes[list_pointer] += 1
        list_pointer = (list_pointer+1) % len(particles)
    return new_particles

class CuriousSimulatedAnnealing(Algorithm): #(n_iter, init_state=None, n_particles=6, temperature_schedule=None)
    def __init__(self, hparams, problem_size, comm, logger) -> None:
        super().__init__(hparams, problem_size, comm, logger)
        self.register_hyperparameter('t0', 100)
        self.register_hyperparameter('popsize', 6)
        self.parse_hyperparameters()

        self.T0 = self.hparams['t0']
        self.popsize = self.hparams['popsize']
        # TODO: current temperature function is hard coded
        self.f = lambda x: 0.9 * x

    def run(self, num_steps, evaluation_session) -> None:
        # Initialize communication
        world_size = self.comm.Get_size()
        my_rank = self.comm.Get_rank()

        # Initialize the particles
        n_particles = self.popsize
        temp = self.T0

        if my_rank == 0:

            init_state = get_random_solution(self.problem_size)
            particles = [init_state for _ in range(n_particles)]
            particle_weights = np.ones(n_particles) / n_particles
            path = [(init_state, init_state.cost(evaluation_session))]

            # Initialize the current state and current energy
            current_state = init_state
            current_energy = init_state.cost(evaluation_session)

            print('Cost= ', current_energy, end=' ')
            current_state.display()

        # Iterate over the temperature schedule
        k = 0
        while k < num_steps/n_particles:
            if my_rank == 0:
                # Resample the particles based on the current weights
                indices = np.random.choice(np.arange(n_particles), size=n_particles, p=particle_weights)
                particles = [particles[i] for i in indices]
                particle_weights = np.ones(n_particles) / n_particles
                particles = group_particles(particles, self.comm.Get_size())
            else:
                particles = None

            # Update each particle
            particles = self.comm.scatter(particles,root=0)
            for i in range(len(particles)):
                # Perturb the particle
                perturbed_particle = particles[i].get_random_neighbor()

                # Calculate the energy difference
                energy_diff = perturbed_particle.cost(evaluation_session) - particles[i].cost(evaluation_session)
                print('N=', my_rank, 'Cost= ', perturbed_particle.cost(evaluation_session), end=' ')
                perturbed_particle.display()

                # Update the particle or move to a new state with a certain probability
                if energy_diff > 0 or acceptance_func(energy_diff, temp) > np.random.uniform():
                    particles[i] = perturbed_particle
            # Update the particle weights based on the new states
            particles = self.comm.gather(particles, root=0)

            if my_rank == 0:
                particles = ungroup_particles(particles)
                for i in range(n_particles):
                    particle_weights[i] = np.exp((particles[i].cost(evaluation_session))/temp)

                # Normalize the weights
                particle_weights /= np.sum(particle_weights)

                # Update the current state and current energy
                best_particle = particles[np.argmax([p.cost(evaluation_session) for p in particles])]
                best_energy = best_particle.cost(evaluation_session)

                if best_energy > current_energy:
                    current_state = best_particle
                    current_energy = best_energy
                    path.append((current_state, current_energy))
                    print('New best:', end=' ')
                    current_state.display()
                    print('Actual Cost: ' + str(current_energy))

            temp = self.f(temp)
            k += 1

        if my_rank != 0:
            return None, None, None
        return current_state, current_energy, path

