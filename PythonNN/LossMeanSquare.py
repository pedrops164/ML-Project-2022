from Loss import Loss
import numpy as np

class LossMeanSquare(Loss):
    def forward(self, pred_values, target_values):
        # pred values and target values have shape (n_samples, n_output)
        subtract = pred_values - target_values
        losses = subtract * subtract
        losses = losses.T

        # losses have shape (n_output, n_samples)
        return losses
