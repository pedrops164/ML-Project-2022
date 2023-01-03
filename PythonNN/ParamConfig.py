class ParamConfig:
    def __init__(self,
        n_hl,
        neurons_per_hl,
        n_it,
        lr,
        lr_decay,
        momentum,
        min_lr
        ):

        self.n_hl = n_hl
        self.neurons_per_hl = neurons_per_hl
        self.n_it = n_it
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.min_lr = min_lr

    def toString(self):
        return \
            "N hidden layers: " + str(self.n_hl) + "\n" + \
            "Neurons per hidden layer: " + str(self.neurons_per_hl) + "\n" + \
            "N epochs: " + str(self.n_it) + "\n" + \
            "Initial learning rate: " + str(self.lr) + "\n" + \
            "Learning rate decay: " + str(self.lr_decay) + "\n" + \
            "Momentum: " + str(self.momentum) + "\n" + \
            "Minimum learning rate: " + str(self.min_lr)


