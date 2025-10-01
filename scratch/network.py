import time
import numpy as np
import scratch.utils as utils
from scratch.lr_scheduler import cosine_annealing


class Network():
    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        self.sizes = sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.activation_func = utils.sigmoid
        self.activation_func_deriv = utils.sigmoid_deriv
        self.output_func = utils.softmax
        self.output_func_deriv = utils.softmax_deriv
        self.cost_func = utils.mse
        self.cost_func_deriv = utils.mse_deriv

        self.params = self._initialize_weights()


    def _initialize_weights(self):
        # number of neurons in each layer
        input_layer = self.sizes[0]
        hidden_layer_1 = self.sizes[1]
        hidden_layer_2 = self.sizes[2]
        output_layer = self.sizes[3]

        # random initialization of weights
        np.random.seed(self.random_state)
        params = {
            'W1': np.random.rand(hidden_layer_1, input_layer) - 0.5,
            'W2': np.random.rand(hidden_layer_2, hidden_layer_1) - 0.5,
            'W3': np.random.rand(output_layer, hidden_layer_2) - 0.5,
        }

        return params


    def _forward_pass(self, x_train):
        '''
        TODO: Implement the forward propagation algorithm.

        The method should return the output of the network.

        The forward pass turns input pixels into class probabilities by stacking:
        - Affine linear function
        - Non-linear activation (sigmoid) for hidden layers
        - Linear output function
        - Softmax converting scores into probabilities

        We will cache everything for backprop
        '''
        X = np.asarray(x_train, dtype=float)
        if X.ndim == 1: # single sample
            X = X.reshape(1, -1)

        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']

        # Layer 1 (Input to Hidden Layer 1)
        Z1 = X @ W1.T
        A1 = self.activation_func(Z1)

        # Layer 2 (Hidden Layer 1 to Hidden Layer 2)
        Z2 = A1 @ W2.T
        A2 = self.activation_func(Z2)

        # Layer 3 (Hidden Layer 2 to Output Layer)
        Z3 = A2 @ W3.T
        Y_hat = self.output_func(Z3)

        # Cache for backward pass
        self.params['X'] = X
        self.params['Z1'], self.params['A1'] = Z1, A1
        self.params['Z2'], self.params['A2'] = Z2, A2
        self.params['Z3'], self.params['Y_hat'] = Z3, Y_hat

        return Y_hat


    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.

        The method should return a dictionary of the weight gradients which are used to update the weights in
        self._update_weights().
        Backpropagation for 2 hidden sigmoid layers + softmax output with MSE loss.
        Returns gradients for W1, W2, W3 (matching shapes of self.params weights).
        '''
        # Ensure 2D (batch-first) shapes
        Y = np.asarray(y_train, dtype=float)
        S = np.asarray(output, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        if S.ndim == 1:
            S = S.reshape(1, -1)

        # Cached forward-pass values
        X = self.params['X']  # (N, D)
        A1 = self.params['A1']  # (N, H1)
        A2 = self.params['A2']  # (N, H2)
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']

        N = X.shape[0]

        # Gradient of Loss w.r.t. Final Layer Input
        # dL/ds for MSE: (s - y)/N
        dL_ds = (S - Y) / N  # (N, K)

        # Gradient of Loss w.r.t. Final Layer Input
        # dL/dz3 via softmax Jacobian: (diag(s) - s s^T) @ dL/ds
        SV = S * dL_ds  # elementwise (N, K)
        s_dot_v = np.sum(SV, axis=1, keepdims=True)  # (N, 1)
        G3 = SV - S * s_dot_v  # (N, K)  -> gradient wrt Z3

        # Backpropagation to Hidden Layers
        # Backprop to hidden layer 2
        sigma2_prime = self.activation_func_deriv(A2)  # (N, H2)
        G2 = (G3 @ W3) * sigma2_prime  # (N, H2)

        # Backprop to hidden layer 1
        sigma1_prime = self.activation_func_deriv(A1)  # (N, H1)
        G1 = (G2 @ W2) * sigma1_prime  # (N, H1)

        # Weight gradients (out, in)
        dW3 = G3.T @ A2  # (K, H2)
        dW2 = G2.T @ A1  # (H2, H1)
        dW1 = G1.T @ X  # (H1, D)

        return {'dW1': dW1, 'dW2': dW2, 'dW3': dW3}

    def _update_weights(self, weights_gradient, learning_rate):
        '''
        TODO: Update the network weights according to stochastic gradient descent.
        Stochastic Gradient Descent (SGD) weight update:
        W <- W - lr * dW
        '''
        self.params['W1'] -= learning_rate * weights_gradient['dW1']
        self.params['W2'] -= learning_rate * weights_gradient['dW2']
        self.params['W3'] -= learning_rate * weights_gradient['dW3']


    def _print_learning_progress(self, start_time, iteration, x_train, y_train, x_val, y_val):
        train_accuracy = self.compute_accuracy(x_train, y_train)
        val_accuracy = self.compute_accuracy(x_val, y_val)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )


    def compute_accuracy(self, x_val, y_val):
        predictions = []
        for x, y in zip(x_val, y_val):
            pred = self.predict(x)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)


    def predict(self, x):
        '''
        TODO: Implement the prediction making of the network.

        This method predicts the class index for the given input `x`.
        Parameters:
        - x: Input data, either a single sample or a batch of samples.

        Returns:
        - int or np.ndarray: Predicted class index/indices.
        '''
        X = np.asarray(x, dtype=float)
        probs = self._forward_pass(X)  # softmax probabilities

        # Single sample: ensure a plain int
        # If `x` is a single sample (D,), it returns the index of the most likely class as an integer.
        if probs.ndim == 2 and probs.shape[0] == 1:
            return int(np.argmax(probs, axis=1)[0])
        # If `x` is a batch of samples (N, D), it returns an array of class indices.
        if probs.ndim == 1:
            return int(np.argmax(probs))

        # Batch: return an array of class indices
        return np.argmax(probs, axis=1)


    def fit(self, x_train, y_train, x_val, y_val, cosine_annealing_lr=False):

        start_time = time.time()

        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                
                if cosine_annealing_lr:
                    learning_rate = cosine_annealing(self.learning_rate, 
                                                     iteration, 
                                                     len(x_train), 
                                                     self.learning_rate)
                else: 
                    learning_rate = self.learning_rate
                output = self._forward_pass(x)
                weights_gradient = self._backward_pass(y, output)
                
                self._update_weights(weights_gradient, learning_rate=learning_rate)

            self._print_learning_progress(start_time, iteration, x_train, y_train, x_val, y_val)
