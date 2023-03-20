class Simulator:
    def __init__(self, logger, run_counter=0, solutions_counter=0) -> None:
        logger.write_info('Simulator initialized')
        self.run_counter = run_counter # counts how many solutions were evaluated in a run
        self.sol_counter = solutions_counter # counts how many solutions were instanced in a run

    def sol_increase(self):
        self.sol_counter = self.sol_counter + 1

    def run_increase(self, num_evaluations):
        self.run_counter = self.run_counter + num_evaluations

    def display(self):
        print(str(self.sol_counter) + ' Solutions have been instanced. The function has been executed ' +
              str(self.run_counter) + ' times.')
