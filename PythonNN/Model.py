from CV import CrossValidation
from CUP_NN import CUP_NN
import numpy as np

class Model:
    def model_selection(self, X, Y, grid, cv, top_n=1):
    
        self.top_n = top_n
        best_model = None
        best_training_error = None
        best_validation_error = None
        current_cfg = 1
        print("Initial configs: ", len(grid.configs))
        configs = remove_bad_configs(CUP_NN, grid.configs, X, Y)
        print("Filtered configs: ", len(configs))

        models = []
        training_errors = []
        validation_errors = []
        for config in configs:
            # config is of type ParamConfig
            # cv.train_config(config, X, Y, n_runs)
            nn, tr_error, vl_error = cv.cross_validation(CUP_NN, config, X, Y)
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
                best_training_error = tr_error
                best_validation_error = vl_error

            print("\nConfig " + str(current_cfg) + " of " + str(len(configs)))
            print("\nParams: " + config.toString())
            print("Training error: " + str(tr_error))
            print("Validation error: " + str(vl_error) + "\n")
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
        self.neural_networks = models[:top_n]
        self.training_error = np.mean(training_errors[:top_n])
        self.validation_error = np.mean(validation_errors[:top_n])
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
        print("Training error: " + str(self.training_error))
        print("Validation error: " + str(self.validation_error))
        print("Test error: " + str(self.test_error))
        print("Final Hyper parameters:\n")
        for i in range(self.top_n):
            print("Model ", i)
            current_nn = self.neural_networks[i]
            print(current_nn.getParamConfig())

    # Given an input, returns the expected output of the trained neural network
    def calculate_output(self, input):
        expected_outputs = []
        for nn in self.neural_networks:
            expected_output = nn.output(input)
            expected_outputs.append(expected_output[0])
        target_output = np.mean(expected_outputs, axis=0)
        return target_output

def remove_bad_configs(nn, config_list, X, Y, loss_threshold = 3.):
    decent_configs = []

    for config in config_list:
        current_nn = nn(config)
        # we are only running part of the training, and checking the loss
        # at that point, to check if it's worth training with this config
        current_nn.n_it = round(0.2 * current_nn.n_it)
        current_nn.train(X,Y,print_progress=False)
        output, loss = current_nn.forward(X, Y)
        if loss < loss_threshold:
            decent_configs.append(config)

    return decent_configs