from network import Network
from layer import FCLayer, ActivationLayer
from activation import sigmoid, sigmoid_prime, relu, relu_prime
from loss import MSE, MSE_prime
from utils import read_data, labels_match
    
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = read_data()

    # create and define the network
    network = Network()
    network.layer(layer=FCLayer(9, 6))
    network.layer(layer=ActivationLayer(sigmoid, sigmoid_prime))
    network.layer(layer=FCLayer(6, 7))
    network.layer(layer=ActivationLayer(relu, relu_prime))

    # train
    # network.disable_logs()
    network.use_loss(MSE, MSE_prime)
    network.train(x_train, y_train, 5000, 0.1)

    # test
    predictions = network.predict(x_test)

    # assert successes
    count = 0
    for i in range(len(predictions)):
        if labels_match(predictions[i], y_test[i]):
            count += 1

    print("test_success_rate: ", count / len(predictions))