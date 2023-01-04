from ParamConfig import ParamConfig

class Grid:
    def __init__(self,
        n_hl_list, # list of numbers of hidden layers
        neurons_per_hl_list, # list of neurons per hidden layer
        n_it_list, # list of numbers of iterations
        lr_list, # list of learning rates
        lr_decay_list, # list of learning rate decays
        momentum_list, # list of momentum values
        min_lr_list, # list of minimum learning rates
        lambda_list, # list of l2 regularization lambda values
        batch_size_list # list of batch sizes
        ):
        self.n_configs = len(n_hl_list) * len(neurons_per_hl_list) * len(n_it_list) * \
            len(lr_list) * len (lr_decay_list) * len(momentum_list) * len(min_lr_list) * \
           len(lambda_list) * len(batch_size_list)
        self.configs = []
        for p1 in n_hl_list:
            for p2 in neurons_per_hl_list:
                for p3 in n_it_list:
                    for p4 in lr_list:
                        for p5 in lr_decay_list:
                            for p6 in momentum_list:
                                for p7 in min_lr_list:
                                    for p8 in lambda_list:
                                        for p9 in batch_size_list:
                                            #add all possible configs to config list
                                            config = ParamConfig(p1,p2,p3,p4,p5,p6,p7,p8,p9)
                                            self.configs.append(config)

