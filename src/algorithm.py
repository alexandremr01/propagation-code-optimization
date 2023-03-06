class Algorithm:
    def __init__(self, hparams, problem_size) -> None:
        self.hparams = hparams
        self.problem_size = problem_size

    def run(self, num_steps) -> None:
        raise NotImplementedError