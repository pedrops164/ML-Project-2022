from ParamConfig import ParamConfig

class Grid:
    def __init__(self,
        n_hl_list,
        neurons_per_hl_list,
        n_it_list,
        lr_list,
        lr_decay_list,
        momentum_list
        ):
        self.configs = []
        for p1 in n_hl_list:
            for p2 in neurons_per_hl_list:
                for p3 in n_it_list:
                    for p4 in lr_list:
                        for p5 in lr_decay_list:
                            for p6 in momentum_list:
                                #add all possible configs to config list
                                config = ParamConfig(p1,p2,p3,p4,p5,p6)
                                self.configs.append(config)

