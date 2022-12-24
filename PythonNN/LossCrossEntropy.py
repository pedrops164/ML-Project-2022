from Loss import Loss
import numpy as np

class LossCrossEntropy(Loss):
    # returns array with loss for each sample
    def forward(self, pred_values, target_values):
        clipped_confidences = np.clip(pred_values, 1e-7, 1-1e-7)
        if len(target_values.shape) == 1:
            #target values is an array of scalar values
            correct_confidences = clipped_confidences[range(target_values.len()), target_values]
        elif len(target_values.shape) == 2:
            #target values is a 2d matrix
            #one hot encoding for each sample
            correct_confidences = np.sum(clipped_confidences*target_values, axis=1)
        return -np.log(correct_confidences)