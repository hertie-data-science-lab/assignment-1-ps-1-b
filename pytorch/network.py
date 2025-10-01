import time

import torch
import torch.nn as nn
import torch.optim as optim


class TorchNetwork(nn.Module):
    def __init__(self, sizes, epochs=10, learning_rate=0.01, random_state=1):
        super().__init__()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        torch.manual_seed(self.random_state)

        self.linear1 = nn.Linear(sizes[0], sizes[1])
        self.linear2 = nn.Linear(sizes[1], sizes[2])
        self.linear3 = nn.Linear(sizes[2], sizes[3])

        self.activation_func = torch.sigmoid
        self.output_func = torch.softmax

        #self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func = nn.MSELoss()

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        self.train_accuracies = []
        self.val_accuracies = []


    def _forward_pass(self, x_train):
        '''
        TODO: Implement the forward propagation algorithm.
        The method should return the output of the network.
        '''

        x = x_train
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.activation_func(self.linear1(x_train))
        x = self.activation_func(self.linear2(x))
        x = self.activation_func(self.linear3(x))
        return self.output_func(x, dim=1)


    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.

        '''
        # Add ´self.optimizer.zero_grad´ to clear gradients from the previous iteration
        # before computing new gradients during backpropagation.
        self.optimizer.zero_grad(set_to_none=True)

        # ChatGPT suggested this change to ensure y_train is a tensor
        if not isinstance(y_train, torch.Tensor):
            y = torch.tensor(y_train)
        else:
            y = y_train
        if y.dim() == 1:
            y = y.unsqueeze(0)
        y = y.to(output.device, dtype=output.dtype)

        loss = self.loss_func(output, y_train.float())
        loss.backward()
        return loss


    def _update_weights(self):
        '''
        TODO: Update the network weights according to stochastic gradient descent.

        Already implemented.
        '''
        self.optimizer.step()


    def _flatten(self, x):
        return x.view(x.size(0), -1)       


    def _print_learning_progress(self, start_time, iteration, train_loader, val_loader):
        train_accuracy = self.compute_accuracy(train_loader)
        val_accuracy = self.compute_accuracy(val_loader)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Learning Rate: {self.optimizer.param_groups[0]["lr"]}, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )


    def predict(self, x):
        '''
        TODO: Implement the prediction making of the network.

        The method should return the index of the most likeliest output class.
        '''
        x = self._flatten(x)
        output = self._forward_pass(x)
        return output.argmax(dim=1)


    def fit(self, train_loader, val_loader):
        start_time = time.time()

        for iteration in range(self.epochs):
            for x, y in train_loader:
                x = self._flatten(x)
                y = nn.functional.one_hot(y, 10)
                self.optimizer.zero_grad()


                output = self._forward_pass(x)
                self._backward_pass(y, output)
                self._update_weights()

            self._print_learning_progress(start_time, iteration, train_loader, val_loader)
            self.train_accuracies.append(self.compute_accuracy(train_loader))
            self.val_accuracies.append(self.compute_accuracy(val_loader))




    def compute_accuracy(self, data_loader):
        correct = 0
        for x, y in data_loader:
            pred = self.predict(x)
            correct += torch.sum(torch.eq(pred, y))

        return correct / len(data_loader.dataset)
