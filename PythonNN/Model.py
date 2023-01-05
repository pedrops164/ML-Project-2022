from CV import CrossValidation
from CUP_NN import CUP_NN
import numpy as np

class Model:
    def model_selection(self, X, Y, grid, cv):
    
        best_model = None
        best_training_error = None
        best_validation_error = None
        current_cfg = 1
        for config in grid.configs:
            # config is of type ParamConfig
            # cv.train_config(config, X, Y, n_runs)
            nn, tr_error, vl_error = cv.cross_validation(CUP_NN, config, X, Y)
            if best_model == None:
                best_model = nn
            if best_training_error == None:
                best_training_error = tr_error
            if best_validation_error == None:
                best_validation_error = vl_error
            
    
            if vl_error < best_validation_error:
                best_model = nn
                best_training_error = tr_error
                best_validation_error = vl_error

            print("\nConfig " + str(current_cfg) + " of " + str(grid.n_configs))
            print("\nParams: " + config.toString())
            print("Training error: " + str(tr_error))
            print("Validation error: " + str(vl_error) + "\n")
            current_cfg += 1
    
        self.neural_network = best_model
        self.training_error = best_training_error
        self.validation_error = best_validation_error
        # model selection returns the best model
    
    def model_assessment(self, test_X, test_Y):
        # having chosen final model, model assessment estimates/evaluates its prediction error 
        # on new test data. Returns an estimation value (loss)
        output, loss = self.neural_network.forward(test_X, test_Y)
        self.test_error = loss

    def print_model(self):
        print("Training error: " + str(self.training_error))
        print("Validation error: " + str(self.validation_error))
        print("Test error: " + str(self.test_error))
        print("Final Hyper parameters:\n" + self.neural_network.getParamConfig())


    # Given an input, returns the expected output of the trained neural network
    def calculate_output(self, input):
        expected_output = self.neural_network.output(input)
        return expected_output[0]