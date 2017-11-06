import network
import mnist_loader

# get training data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# make network
# sizes: input(784), hidden(30), output(10)
net = network.Network([784, 30, 10])

# train network using stochastic gradient descent
# over 30 epochs, with a mini-batch size of 10, and a learning rate of n=3.0
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# exercise:
# network with just two layers, input and output (w/ 784 and 10 neurons respectively)
# train using stochastic gradient descent
net = network.Network([784, 0, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
