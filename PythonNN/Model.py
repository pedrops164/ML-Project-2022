from CV import CrossValidation
import numpy as np

class Model:
    def model_selection(self, X, Y, grid, cv):
    
        best_model = None
        best_training_error = None
        best_validation_error = None
        for config in grid.configs:
            # config is of type ParamConfig
            nn, tr_error, vl_error = cv.cross_validation(config, X, Y)
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
    
        self.neural_network = best_model
        self.training_error = best_training_error
        self.validation_error = best_validation_error
        # model selection returns the best model
    
    def model_assessment(self, test_X, test_Y):
        # having chosen final model, model assessment estimates/evaluates its prediction error 
        # on new test data. Returns an estimation value (loss)
        seperate_loss = self.neural_network.forward(test_X, test_Y)
        self.test_error = np.mean(seperate_loss)

    def print_model(self):
        print("Training error: " + str(self.training_error))
        print("Validation error: " + str(self.validation_error))
        print("Test error: " + str(self.test_error))
        print("Final Hyper parameters:\n" + self.neural_network.getParamConfig())
