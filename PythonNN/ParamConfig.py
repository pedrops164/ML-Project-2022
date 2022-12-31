class ParamConfig:
    def __init__(self,
        n_hl,
        neurons_per_hl,
        n_it,
        lr,
        lr_decay,
        momentum
        ):

        self.n_hl = n_hl
        self.neurons_per_hl = neurons_per_hl
        self.n_it = n_it
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum

