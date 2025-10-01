import time
import numpy as np
from scratch.network import Network

class ResNetwork(Network):


    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        '''
        TODO: Initialize the class inheriting from scratch.network.Network.
        The method should check whether the residual network is properly initialized.
        '''
        super().__init__(sizes, epochs, learning_rate, random_state)

        # Initialize the projection matrix W_proj for input to hidden layer 1
        self.params['W_proj'] = np.random.randn(sizes[0], sizes[1])  # Project input to the first hidden layer size
        # Initialize the projection matrix W_proj2 for residual connection between A1_residual and A2
        self.params['W_proj2'] = np.random.randn(sizes[1], sizes[2])  # Project A1_residual to match A2 size



    def _forward_pass(self, x_train):
        '''
        TODO: Implement the forward propagation algorithm.
        The method should return the output of the network.
        '''

        X = np.asarray(x_train, dtype=float)
        if X.ndim == 1:  # single sample
            X = X.reshape(1, -1)

        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        W_proj = self.params['W_proj']  # Projection matrix for input to first hidden layer
        W_proj2 = self.params['W_proj2']  # Projection matrix for residual connection between A1_residual and A2

        # Layer 1 (Input to Hidden Layer 1)
        Z1 = np.clip(X @ W1.T, -500, 500)  # Clipping to prevent overflow
        A1 = self.activation_func(Z1)

        # Residual connection: Add input (projected) to the output of layer 1
        X_proj = X @ W_proj  # Project input to the same space as A1
        A1_residual = A1 + X_proj  # Add the projected input to the output (residual)

        # Layer 2 (Hidden Layer 1 to Hidden Layer 2)
        Z2 = np.clip(A1_residual @ W2.T, -500, 500)  # Clipping to prevent overflow
        A2 = self.activation_func(Z2)

        # Residual connection: Add output of layer 1 to layer 2
        # Project A1_residual to the size of A2 using W_proj2
        A1_residual_proj = A1_residual @ W_proj2  # Project A1_residual to match the size of A2
        A2_residual = A2 + A1_residual_proj  # Adding the residual connection

        # Layer 3 (Hidden Layer 2 to Output Layer)
        Z3 = A2_residual @ W3.T  # Need the transpose for correct shape!

        # Stability adjustment for softmax
        Z3 = Z3 - np.max(Z3, axis=1, keepdims=True)  # Stability adjustment for softmax
        Y_hat = self.output_func(Z3.T).T  # Transpose in/out to match the shape

        # Cache for backward pass
        self.params['X'] = X
        self.params['Z1'], self.params['A1'] = Z1, A1
        self.params['Z2'], self.params['A2'] = Z2, A2
        self.params['Z3'], self.params['Y_hat'] = Z3, Y_hat
        self.params['A1_residual'] = A1_residual
        self.params['A2_residual'] = A2_residual

        return Y_hat



    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.
        The method should return a dictionary of the weight gradients which are used to update the weights in self._update_weights().
        The method should also account for the residual connection in the hidden layer.

        '''
        Y = np.asarray(y_train, dtype=float)
        S = np.asarray(output, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        if S.ndim == 1:
            S = S.reshape(1, -1)

        X = self.params['X']
        A1_residual = self.params['A1_residual']
        A2_residual = self.params['A2_residual']
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']

        N = X.shape[0]

        # Gradient of Loss w.r.t. Final Layer Input (softmax + MSE)
        dL_ds = (S - Y) / N  # Elementwise difference (s - y) / N

        # Gradient of Loss w.r.t. Final Layer Input (softmax Jacobian)
        SV = S * dL_ds  # Elementwise product (N, K)
        s_dot_v = np.sum(SV, axis=1, keepdims=True)  # Sum over classes (N, 1)
        G3 = SV - S * s_dot_v  # Gradient w.r.t. Z3 (N, K)

        # Backpropagation to Hidden Layers
        # Sigmoid derivatives (safe: uses activations)
        sigma2_prime = A2_residual * (1.0 - A2_residual)  # Sigmoid derivative for layer 2 (N, H2)
        G2 = (G3 @ W3) * sigma2_prime  # Gradient w.r.t. A2 (N, H2)

        sigma1_prime = A1_residual * (1.0 - A1_residual)  # Sigmoid derivative for layer 1 (N, H1)
        G1 = (G2 @ W2) * sigma1_prime  # Gradient w.r.t. A1 (N, H1)

        # Weight gradients
        dW3 = G3.T @ A2_residual  # Gradient w.r.t. W3
        dW2 = G2.T @ A1_residual  # Gradient w.r.t. W2
        dW1 = G1.T @ X  # Gradient w.r.t. W1

        # Return gradients
        return {'dW1': dW1, 'dW2': dW2, 'dW3': dW3}
