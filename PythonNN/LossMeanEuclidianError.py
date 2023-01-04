from Loss import Loss
import numpy as np

class MEE(Loss):
    def forward(self, pred_y, true_y):
        # predicted values and target values have shape (n_samples, n_output)
        subtract = pred_y - true_y
        losses = subtract * subtract
        sum_losses = np.sum(losses, axis=1)
        sqrt_losses = np.sqrt(sum_losses)

        # we save this value for later, because we use it in backprop
        self.useful_backprop = sqrt_losses

        return sqrt_losses

    def backprop(self, Y):
        n_samples = Y.shape[0]
        nominator = self.output - Y
        loss_gradient = np.divide(nominator.T, self.useful_backprop.T).T
        # loss_gradient = nominator / self.useful_backprop
        self.inputs_deriv = loss_gradient / n_samples


