class Algorithm:
    def __init__(self, hparams, problem_size, logger) -> None:
        self.hparams = hparams
        self.problem_size = problem_size
        self.logger = logger

    def run(self, num_steps) -> None:
        raise NotImplementedError