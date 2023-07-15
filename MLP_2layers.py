# MLP with 2 hidden layers
import numpy as np
from sklearn.base import BaseEstimator

from MLP import compute_loss_and_acc, int_to_onehot, minibatch_generator, sigmoid


class NeuralNetMLP_2layers(BaseEstimator):
    def __init__(self, num_features, num_hidden, num_hidden2, num_classes, random_seed=33):
        """
        :param: num_features: number of features in input layer.
        :param: num_hidden: number of neurons in first hidden layer.
        :param: num_hidden2: number of neurons in second hidden layer.
        :param: num_classes: number of classes, neurons in output layer.
        """
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.random_seed = random_seed
        rng = np.random.RandomState(self.random_seed)
        self.num_hidden = num_hidden
        self.num_hidden2 = num_hidden2
        self.weight_h1 = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.weight_h2 = rng.normal(loc=0.0, scale=0.1, size=(num_hidden2, num_hidden))
        self.bias_h1 = np.zeros(num_hidden)
        self.bias_h2 = np.zeros(num_hidden2)
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden2))
        self.bias_out = np.zeros(num_classes)
        self.epoch_train_acc, self.epoch_train_loss = [], []

    def forward(self, x):
        z_h1 = np.dot(x, self.weight_h1.T) + self.bias_h1
        a_h1 = sigmoid(z_h1)  # output dim: [n_examples, n_hidden]
        z_h2 = np.dot(a_h1, self.weight_h2.T) + self.bias_h2
        a_h2 = sigmoid(z_h2)  # output dim: [n_examples, n_hidden2]
        z_out = np.dot(a_h2, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)  # output dim: [n_examples, n_classes]
        return (a_h1, a_h2), a_out

    def backward(self, x, a_h1, a_h2, a_out, y):
        # Part 1: dLoss/dOutWeights = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        # where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet for convenient re-use
        y_onehot = int_to_onehot(y, self.num_classes)  # onehot encoding
        # input/output dim: [n_examples, n_classes]
        d_loss__d_a_out = 2. * (a_out - y_onehot) / y.shape[0]
        # input/output dim: [n_examples, n_classes]
        d_a_out__d_z_out = a_out * (1. - a_out)  # sigmoid derivative
        # output dim: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z_out  # "delta (rule) placeholder"
        # gradient for output weights
        d_z_out__dw_out = a_h2  # [n_examples, n_hidden]
        # input dim: [n_classes, n_examples] dot [n_examples, n_hidden2]
        # output dim: [n_classes, n_hidden2]
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        # Part 2: dLoss/dHiddenWeights = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet *
        # dHiddenNet/dWeight
        d_z_out__a_h2 = self.weight_out  # [n_classes, n_hidden2]
        d_loss__a_h2 = np.dot(delta_out, d_z_out__a_h2)  # output dim: [n_examples, n_hidden2]
        # [n_examples, n_hidden2]
        d_a_h2__d_z_h2 = a_h2 * (1. - a_h2)  # sigmoid derivative
        # output dim: [n_examples, n_hidden2]
        delta_2 = d_loss__a_h2 * d_a_h2__d_z_h2  # "delta (rule) placeholder"
        d_z_h2__d_w_h2 = a_h1  # [n_examples, n_hidden]
        # output dim: [n_hidden2, n_hidden]
        d_loss__d_w_h2 = np.dot(delta_2.T, d_z_h2__d_w_h2)
        d_loss__d_b_h2 = np.sum(delta_2, axis=0)

        # Part 3
        d_z_2__a_h1 = self.weight_h2  # [n_hidden2, n_hidden]
        # output dim: [n_examples, n_hidden]
        d_loss__a_h1 = np.dot(delta_2, d_z_2__a_h1)
        # [n_examples, n_hidden]
        d_a_h1__d_z_h1 = a_h1 * (1. - a_h1)  # sigmoid derivative
        d_z_h1__d_w_h1 = x  # [n_examples, n_hidden]
        # output dim: [n_hidden2, n_hidden]
        d_loss__d_w_h1 = np.dot((d_loss__a_h1 * d_a_h1__d_z_h1).T, d_z_h1__d_w_h1)
        d_loss__d_b_h1 = np.sum((d_loss__a_h1 * d_a_h1__d_z_h1), axis=0)

        return (d_loss__dw_out, d_loss__db_out,
                d_loss__d_w_h2, d_loss__d_b_h2,
                d_loss__d_w_h1, d_loss__d_b_h1)

    def update_weights(self, X_train_mini, y_train_mini, a_h1, a_h2, a_out, learning_rate):
        """
        Compute gradients and update weights.
        """
        d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h2, d_loss__d_b_h2, \
            d_loss__d_w_h1, d_loss__d_b_h1 = self.backward(X_train_mini, a_h1, a_h2, a_out,
                                                           y_train_mini)
        self.weight_h1 -= learning_rate * d_loss__d_w_h1
        self.bias_h1 -= learning_rate * d_loss__d_b_h1
        self.weight_h2 -= learning_rate * d_loss__d_w_h2
        self.bias_h2 -= learning_rate * d_loss__d_b_h2
        self.weight_out -= learning_rate * d_loss__d_w_out
        self.bias_out -= learning_rate * d_loss__d_b_out

    def epoch_log(self, X_train, y_train, epoch, num_epochs):
        train_loss, train_acc = compute_loss_and_acc(self, X_train, y_train)
        train_acc = train_acc * 100
        self.epoch_train_acc.append(train_acc)
        self.epoch_train_loss.append(train_loss)
        print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
              f'| Train loss: {train_loss:.2f} '
              f'| Train accuracy: {train_acc:.2f}% ')

    def fit(self, X, y, num_epochs=50, minibatch_size=100, learning_rate=0.1):
        for epoch in range(num_epochs):
            minibatch_gen = minibatch_generator(X, y, minibatch_size)
            for X_train_mini, y_train_mini in minibatch_gen:  # iterate over mini batches
                (a_h1, a_h2), a_out = self.forward(X_train_mini)  # outputs
                self.update_weights(X_train_mini, y_train_mini, a_h1, a_h2, a_out, learning_rate)
            self.epoch_log(X, y, epoch, num_epochs)
        (self.a_h1, self.a_h2), self.a_out = self.forward(X)
        self.is_fitted_ = True
        return self
