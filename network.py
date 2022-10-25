from layer import Layer
from utils import shuffle, labels_match

class Network():
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.logs_enabled = True

    def disable_logs(self):
        self.logs_enabled = False

    def layer(self, layer: Layer):
        """
            Adds a Layer to the model.
        """
        self.layers.append(layer)

    def use_loss(self, loss, loss_prime):
        """
            Sets the loss function and its derivative.
        """
        self.loss = loss
        self.loss_prime = loss_prime

    def train(self, x_train, y_train, epochs, learning_rate):
        """
            Trains the network using x_train and y_train.
        """
        prev_err = float('inf')
        k = 0
        update_lr_epochs = 100
        steps = 95

        for i in range(epochs):
            err = 0
            successes = 0

            # shuffle examples (avoid memorizing)
            x_train, y_train = shuffle(x_train, y_train)
            
            for j in range(len(x_train)):
                # forward pass
                output = x_train[j]

                for layer in self.layers:
                    output = layer.forward_pass(output)

                # assert success
                if labels_match(output, y_train[j]):
                    successes += 1
                
                # compute loss
                err += self.loss(y_train[j], output)

                # backward pass
                error = self.loss_prime(y_train[j], output)

                for layer in reversed(self.layers):
                    error = layer.backward_pass(error, learning_rate)

            # average error
            successes /= len(x_train)
            if (i+1) % 100 == 0 and self.logs_enabled:
                print("epoch %d/%d, train_success_rate: %.2f" % (i+1, epochs, successes))

            if err < prev_err:
                k += 1
            prev_err = err

            if (i+1) % update_lr_epochs == 0:
                # save network state
                if k == 0:
                    for i in range(len(self.layers)):
                        if i % 2 == 0:
                            self.layers[i].save_weights()
                
                if k >= steps:
                    learning_rate += 0.01

                    for i in range(len(self.layers)):
                        if i % 2 == 0:
                            self.layers[i].save_weights()
                # restore network state
                else:
                    learning_rate += -0.05 * learning_rate
                    for i in range(len(self.layers)):
                        if i % 2 == 0:
                            self.layers[i].restore_weights()
                
                k = 0

    def predict(self, input_data):
        """
            Returns a prediction for a given input.
        """
        result = []

        for i in range(len(input_data)):
            # forward pass
            output = input_data[i]

            for layer in self.layers:
                output = layer.forward_pass(output)

            result.append(output)

        return result
