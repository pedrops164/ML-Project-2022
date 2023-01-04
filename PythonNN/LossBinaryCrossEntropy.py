from Loss import Loss
import numpy as np

class BCE(Loss):
    # returns array with loss for each sample
    def forward(self, predicted_y, true_y):
        #Log loss = 1/N * sum(-(y_i * log(p_i) + (1-y_i) * log(1-p_i)))

        # number of samples
        n_samples = len(true_y)

        self.inputs = predicted_y

        # we remove the values that are 1 or 0, because the log would give infinity,
        # and thus the mean would give infinity aswell.
        predicted_y = np.clip(predicted_y, 0.0000001, 0.9999999)

        # formula above
        loss = -(true_y * np.log(predicted_y) + (1-true_y) * np.log(1-predicted_y))
        return np.mean(loss, axis=-1)

    def backprop(self, Y):
        n_examples = Y.shape[0]
        n_outputs = Y.shape[1]

        # dL/da = (-true_y/a + (1-true_y)/(1-a))
        # a are the inputs

        a = np.clip(self.inputs, 0.0000001, 0.9999999)

        d = (-Y/a + (1-Y)/(1-a)) / n_outputs

        # here we normalize the derivative, dividing by the number of examples given
        self.inputs_deriv = d / n_examples


