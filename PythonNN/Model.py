from CV import CrossValidation
from CUP_NN import CUP_NN
from PlotMaker import make_plot
import numpy as np
from Grid import Grid

class Model:
    def __init__(self, file):
        self.logfile = file

    def model_selection(self, X, Y, test_X, test_Y, configs, cv, top_n=1):
    
        self.top_n = top_n
        best_model = None
        best_config = None
        best_training_error = None
        best_validation_error = None
        self.logfile.write("Initial configs: " + str(len(configs)) + "\n")
        configs = remove_bad_configs(CUP_NN, configs, X, Y)
        n_configs = len(configs)
        self.logfile.write("Filtered configs: " + str(n_configs) + "\n")
        if self.top_n > n_configs:
            self.top_n = n_configs

        models = []
        training_errors = []
        validation_errors = []
        file_path = "outputs/plot"

        # This for cycle does a grid search with all the parameter configs, and chooses the one
        # with the least validation error
        for config in configs:
            # config is of type ParamConfig

            # we dont want the plots of these configs
            nn, tr_error, vl_error = cv.cross_validation(CUP_NN, config, X, Y)
            if best_training_error == None:
                best_training_error = tr_error
            if best_validation_error == None:
                best_validation_error = vl_error
            
    
            if vl_error < best_validation_error:
                best_config = config
                best_training_error = tr_error
                best_validation_error = vl_error

        self.logfile.write("\nParams of best config: \n" + best_config.toString() + "\n")
        self.logfile.write("Training error: " + str(best_training_error) + "\n")
        self.logfile.write("Validation error: " + str(best_validation_error) + "\n")

        fine_search_configs = create_fine_grid(best_config, 0.0025, 0.00005)

        current_cfg = 1

        self.logfile.write("Now trying out the finer configs...\n")
        # this for cycle goes through all the finer configs, created from the best config, and
        # chooses the best top n configs, so that afterwards we can ensemble.
        for config in fine_search_configs:
            # config is of type ParamConfig
            path = file_path + str(current_cfg) + ".png"
            nn, tr_error, vl_error = cv.cross_validation(CUP_NN, config, X, Y, test_X, test_Y, plot_file_path=path)
            if best_model == None:
                best_model = nn
            if best_training_error == None:
                best_training_error = tr_error
            if best_validation_error == None:
                best_validation_error = vl_error


            models.append(nn)
            training_errors.append(tr_error)
            validation_errors.append(vl_error)
            
    
            if vl_error < best_validation_error:
                best_model = nn
                best_config = config
                best_training_error = tr_error
                best_validation_error = vl_error

            self.logfile.write("\nConfig " + str(current_cfg) + " of " + str(len(fine_search_configs)) + "\n")
            self.logfile.write("\nParams: " + config.toString() + "\n")
            self.logfile.write("Training error: " + str(tr_error) + "\n")
            self.logfile.write("Validation error: " + str(vl_error) + "\n")
            current_cfg += 1

        models = np.array(models)
        training_errors = np.array(training_errors)
        validation_errors = np.array(validation_errors)

        # here we sort the models by validation error
        best_vl_order = np.argsort(validation_errors)
        # we set the order on the model array
        models = models[best_vl_order]
        training_errors = training_errors[best_vl_order]
        validation_errors = validation_errors[best_vl_order]
    
        # here we pick the top n models
        self.training_errors = training_errors[:top_n]
        self.validation_errors = validation_errors[:top_n]
        self.neural_networks = models[:top_n]
        self.training_error = np.mean(self.training_errors)
        self.validation_error = np.mean(self.validation_errors)

        # final model learning curves
        final_model_lc_TR = []
        final_model_lc_TS = []
        for nn in self.neural_networks:
            final_model_lc_TR.append(nn.tr_loss_lc)
            final_model_lc_TS.append(nn.ts_loss_lc)

        final_model_lc_TR = np.mean(final_model_lc_TR, axis=0)
        final_model_lc_TS = np.mean(final_model_lc_TS, axis=0)

        make_plot(range(final_model_lc_TR.shape[0]), final_model_lc_TR, final_model_lc_TS, "MEE", "Epochs vs MEE", "outputs/final_model.png", ylim=(0,3))


        # model selection returns the best model
    
    def model_assessment(self, test_X, test_Y):
        # having chosen final model, model assessment estimates/evaluates its prediction error 
        # on new test data. Returns an estimation value (loss)
        outputs = []
        losses = []
        for i in range(self.top_n):
            current_nn = self.neural_networks[i]
            output, loss = current_nn.forward(test_X, test_Y)
            outputs.append(output)
            losses.append(loss)

        self.test_error = np.mean(losses)

    def print_model(self):
        self.logfile.write("Training error: " + str(self.training_error) + "\n")
        self.logfile.write("Validation error: " + str(self.validation_error) + "\n")
        self.logfile.write("Test error: " + str(self.test_error) + "\n")
        self.logfile.write("Final Hyper parameters:\n\n")
        for i in range(self.top_n):
            self.logfile.write("Model " + str(i) + "\n")
            self.logfile.write("Training error: " + str(self.training_errors[i]) + "\n")
            self.logfile.write("Validation error: " + str(self.validation_errors[i]) + "\n")
            current_nn = self.neural_networks[i]
            self.logfile.write(current_nn.getParamConfig() + "\n")

    # Given an input, returns the expected output of the trained neural network
    def calculate_output(self, input):
        expected_outputs = []
        for nn in self.neural_networks:
            expected_output = nn.output(input)
            expected_outputs.append(expected_output[0])
        target_output = np.mean(expected_outputs, axis=0)
        return target_output

    def reset_params(self):
        for nn in self.neural_networks:
            nn.reset_params()

    # we retrain the best models on the whole dataset
    def retrain(self, X, Y):
        for nn in self.neural_networks:
            nn.train(X,Y,print_progress=False)

def remove_bad_configs(nn, config_list, X, Y, loss_threshold = 100.):
    decent_configs = []

    for config in config_list:
        current_nn = nn(config)
        # we are only running part of the training, and checking the loss
        # at that point, to check if it's worth training with this config
        current_nn.n_it = round(0.05 * current_nn.n_it)
        current_nn.train(X,Y,print_progress=False)
        output, loss = current_nn.forward(X, Y)
        if loss < loss_threshold:
            decent_configs.append(config)

    return decent_configs

"""
This function receives a param config object, and returns a list of configs that slightly tweak some
hyperparameters
"""
def create_fine_grid(pg, lr_gap=0.005, lambda_param_gap=0.00005):
    new_lr_list = np.random.uniform(pg.lr - lr_gap, pg.lr + lr_gap, 2)
    new_lambda_list = np.random.uniform(pg.lambda_param - lambda_param_gap, pg.lambda_param + lambda_param_gap, 2)

    grid = Grid([pg.n_hl],
                [pg.neurons_per_hl],
                [pg.n_it],
                new_lr_list,
                [pg.lr_decay],
                [pg.momentum],
                [pg.min_lr],
                new_lambda_list,
                [pg.batch_size])


    return grid.configs