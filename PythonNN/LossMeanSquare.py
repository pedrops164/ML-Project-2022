from Loss import Loss
import numpy as np

class MSE(Loss):
    def forward(self, pred_values, target_values):
        # pred values and target values have shape (n_samples, n_output)
        subtract = pred_values - target_values
        self.useful_backprop = subtract
        losses = subtract * subtract
        sum_losses = np.sum(losses, axis=1)

        return sum_losses

    def backprop(self, Y):
        n_samples = Y.shape[0]
        loss_gradient = 2 * self.useful_backprop
        self.inputs_deriv = loss_gradient / n_samples
