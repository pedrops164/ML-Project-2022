import numpy as np

class Loss:
    def calculate(self, output, y):
        self.output = output
        losses = self.forward(output, y) # returns 1d array
        return np.mean(losses)

    def title(self):
        return "Loss"

class BCE(Loss):
    # returns array with loss for each sample
    def forward(self, predicted_y, true_y):
        #Log loss = 1/N * sum(-(y_i * log(p_i) + (1-y_i) * log(1-p_i)))

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

    def title(self):
        return "MEE"


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

    def title(self):
        return "MSE"