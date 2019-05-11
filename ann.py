###############################################################################
# The data being fed while training must be flattened                         #
# The code does not handle any preprocessing part on the features and targets #
###############################################################################

# import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from keras.datasets import mnist

# define a class for the Artificial Neural Network (Basic Neural Network)
class ANN:

    # define the constructor for initialization of random weights and bias
    def __init__(self, neurons, activationList):
        # initialize the arrays for weights and biases
        self.weights = np.array([np.random.randn(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))])
        self.bias = np.array([np.random.randn(neurons[i]) for i in range(1, len(neurons))])

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
        weightedSum = None
        # feed the input x through all the layers one by one
        for i in range(self.numOfLayers):
            weightedSum = np.add(np.dot(activation, self.weights[i]), self.bias[i])
            activation = self.activationList[i](weightedSum)
        return activation

    # define the function for nudging (optimizing) the weights and biases
    def train(self, x, y, learnRate, epochs, batchSize):
        # define a list for storing the cost at given number of iterations
        costList = []
        
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
                temp_cost = (cost ** 2).sum() / batchSize
                costList.append(temp_cost)
                print("Epoch", e + 1, "|", "Batch", b / batchSize, "|", "Cost", temp_cost)

                # perform backpropagation
                i = self.numOfLayers
                while i > 0:
                    cost = cost * self.activationDashList[i - 1](weightedSum[i])
                    db[i - 1] = cost.sum(axis=0) / batchSize
                    dw[i - 1] = np.matmul(activation[i - 1].T, cost) / batchSize
                    cost = np.matmul(cost, self.weights[i - 1].T)
                    i = i - 1

                # convert the lists to arrays
                dw = np.array(dw)
                db = np.array(db)

                # nudge the weights and biases
                self.weights = self.weights - learnRate * dw
                self.bias = self.bias - learnRate * db

        # plot the learning curve
        plt.plot(range(1, len(costList) + 1), costList)
        plt.scatter(range(1, len(costList) + 1), costList)        
        plt.show()
            
    # define the activation functions and the derivative functions
    @staticmethod
    def sigmoid(x):
        temp = 1.0 / (1.0 + np.exp(-x))
        return temp

    @staticmethod
    def sigmoidDash(x):
        temp = (1.0 / (1.0 + np.exp(-x))) * (1.0 - (1.0 / (1.0 + np.exp(-x))))
        return temp
    
    @staticmethod
    def relu(x):
        temp = x.copy()
        for i in range(len(temp)):
            for j in range(len(temp[i])):
                if temp[i][j] < 0:
                    temp[i][j] = 0.01 * temp[i][j]
        # return x * (x > 0)
        return temp

    @staticmethod
    def reluDash(x):
        temp = x.copy()
        for i in range(len(temp)):
            for j in range(len(temp[i])):
                if temp[i][j] < 0:
                    temp[i][j] = 0.01
                elif temp[i][j] > 0:
                    temp[i][j] = 1
                elif temp[i][j] == 0:
                    temp[i][j] = 0
        # return x > 0
        return temp

    @staticmethod
    def tanh(x):
        temp = np.tanh(x)
        return temp

    @staticmethod
    def tanhDash(x):
        temp = 1.0 - tanh(x) ** 2
        return temp

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linearDash(x):
        return 1
