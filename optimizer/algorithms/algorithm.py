class Algorithm:
    def __init__(self, hparams, problem_size, comm, logger, optimize_problem_size) -> None:
        self.default_hparams = { }
        self.hparams = { } # this will remain empty untill parse_hyperparameters
        self.raw_hparams = hparams
        self.problem_size = problem_size
        self.hyperparameters_parsed = False
        self.comm = comm
        self.logger = logger
        self.optimize_problem_size = optimize_problem_size

    def run(self, num_steps, evaluation_session) -> None:
        raise NotImplementedError
    
    def register_hyperparameter(self, key, default_value):
        if self.hyperparameters_parsed:
            raise Exception('Hyperparameters already parsed')
        self.default_hparams[key] = default_value

    def parse_hyperparameters(self):
        parsed_hparams = self.default_hparams.copy()
        for k, v in self.raw_hparams.items():
            if self.default_hparams.get(k) is None:
                raise Exception(f'Hyperparameter {k} not registered')
            parsed_hparams[k] = v
        self.hparams = parsed_hparams
        self.hyperparameters_parsed = True
            
