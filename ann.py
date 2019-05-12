# import the required libraries
import numpy as np
import matplotlib.pyplot as plt

# define a class for the Artificial Neural Network (Basic Neural Network)
class ANN:

    # define the constructor for initialization of random weights and bias
    def __init__(self, neurons, activationList):
        # initialize the arrays for weights and biases
        self.weights = [np.random.randn(neurons[i - 1], neurons[i]) / 1000 for i in range(1, len(neurons))]
        self.bias = [np.random.randn(neurons[i]) / 1000 for i in range(1, len(neurons))]

        # store the list which has the activation function to be applied at the end of each layer
        self.activationList = activationList
        self.numOfLayers = len(activationList)

        # create a new list that has the derivative of the corresponding activation function for each layer
        self.activationDashList = []
        for func in self.activationList:
            if func == self.sigmoid:
                self.activationDashList.append(self.sigmoidDash)
            elif func == self.relu:
                self.activationDashList.append(self.reluDash)
            elif func == self.tanh:
                self.activationDashList.append(self.tanhDash)
            elif func == self.linear:
                self.activationDashList.append(self.linearDash)

    # define the function for predicting the result given the feature matrix
    def predict(self, x):
        activation = x.copy()
        
        #normalize the inputs
        mu = activation.sum(axis = 0) / float(len(activation))
        activation = activation - mu
        sigma_sq = (activation ** 2).sum(axis = 0) / float(len(x))
        activation = activation * ((sigma_sq + 0.00000001) ** -0.5)
        
        weightedSum = None

        # feed the input x through all the layers one by one
        for i in range(self.numOfLayers):
            weightedSum = np.add(np.matmul(activation, self.weights[i]), self.bias[i])
            activation = self.activationList[i](weightedSum)
        return activation
    
    # define the function for nudging (optimizing) the weights and biases
    def train(self, x, y, learnRate, epochs, batchSize, regParameter):
        # define a list for storing the cost at given number of iterations
        costList = []

        # normalize the inputs
        mu = x.sum(axis = 0) / float(len(x))
        x = x - mu
        sigma_sq = (x ** 2).sum(axis = 0) / float(len(x))
        x = x * ((sigma_sq + 0.00000001) ** -0.5)
        
        # perform the optimization for epochs number of times on the entire dataset
        for e in range(epochs):
            # optimize the weights and biases for every batch
            for b in range(0, len(x), batchSize):
                dw = [None] * self.numOfLayers
                db = [None] * self.numOfLayers
                weightedSum = [None] * (self.numOfLayers + 1)
                activation = [None] * (self.numOfLayers + 1)

                # copy the input activations
                activation[0] = x[b : b + batchSize].copy()
                
                # find the prediction and store the intermediate weighted sums and activations
                for i in range(self.numOfLayers):
                    weightedSum[i + 1] = np.add(np.matmul(activation[i], self.weights[i]), self.bias[i])
                    activation[i + 1] = self.activationList[i](weightedSum[i + 1])

                # calculate the cost
                cost = activation[self.numOfLayers] - y[b : b + batchSize]

                # add the cost for the current batch to the list to plot the learning curve later
                temp_cost = ((cost ** 2) / batchSize).sum()
                costList.append(temp_cost)
                print("Epoch", e + 1, "|", "Batch", b / batchSize, "|", "Cost", temp_cost)

                # perform backpropagation
                i = self.numOfLayers
                while i > 0:
                    cost = cost * self.activationDashList[i - 1](weightedSum[i])
                    db[i - 1] = (cost / batchSize).sum(axis = 0)
                    dw[i - 1] = np.matmul(activation[i - 1].T, cost / batchSize)
                    cost = np.matmul(cost, self.weights[i - 1].T)
                    i -= 1

                # update the weights and biases
                for i in range(self.numOfLayers):
                    # add the regularization term for weights
                    dw[i] += (regParameter / batchSize) * self.weights[i]
                    # nudge the weights and biases
                    self.weights[i] -= learnRate * dw[i]
                    self.bias[i] -= learnRate * db[i]
                
        # plot the learning curve
        plt.plot(range(1, len(costList) + 1), costList)
        plt.scatter(range(1, len(costList) + 1), costList)        
        plt.show()
            
    # define the sigmoid activation function
    @staticmethod
    def sigmoid(x):
        temp = 1.0 / (1.0 + np.exp(-x))
        return temp

    # define the derivative of sigmoid activation function
    @staticmethod
    def sigmoidDash(x):
        temp = (1.0 / (1.0 + np.exp(-x))) * (1.0 - (1.0 / (1.0 + np.exp(-x))))
        return temp

    # define the relu activation function
    @staticmethod
    def relu(x):
        return x * (x > 0)

    # define the derivative of the relu activation function
    @staticmethod
    def reluDash(x):
        return x > 0

    # define the tanh activation function
    @staticmethod
    def tanh(x):
        temp = np.tanh(x)
        return temp

    # define the derivative of the tanh activation function
    @staticmethod
    def tanhDash(x):
        temp = 1.0 - tanh(x) ** 2
        return temp

    # define the linear activation function
    @staticmethod
    def linear(x):
        return x

    # define the derivative of the linear activation function
    @staticmethod
    def linearDash(x):
        return 1
