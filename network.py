from layer import Layer

class Network():
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

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

    def train(self):
        """
            Trains the network using x_train and y_train.
        """
        pass

    def predict(self):
        """
            Returns a prediction for a given input.
        """
        pass
