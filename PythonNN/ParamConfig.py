class ParamConfig:
    def __init__(self,
        n_hl, # number of hidden layers
        neurons_per_hl, # neurons per hidden layer
        n_it, # number of iterations
        lr, # initial learning rate
        lr_decay, # learning rate decay
        momentum, # momentum value
        min_lr, # minimum learning rate
        lambda_param, # l2 regularization lambda value
        batch_size # batch size
        ):

        self.n_hl = n_hl
        self.neurons_per_hl = neurons_per_hl
        self.n_it = n_it
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.min_lr = min_lr
        self.lambda_param = lambda_param
        self.batch_size = batch_size

    def toString(self):
        return \
            "N hidden layers: " + str(self.n_hl) + "\n" + \
            "Neurons per hidden layer: " + str(self.neurons_per_hl) + "\n" + \
            "N epochs: " + str(self.n_it) + "\n" + \
            "Initial learning rate: " + str(self.lr) + "\n" + \
            "Learning rate decay: " + str(self.lr_decay) + "\n" + \
            "Momentum: " + str(self.momentum) + "\n" + \
            "Minimum learning rate: " + str(self.min_lr) + "\n" + \
            "Lambda: " + str(self.lambda_param) + "\n" + \
            "Batch size: " + str(self.batch_size)

