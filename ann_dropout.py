# import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from keras.datasets import fashion_mnist

# define a class for the Artificial Neural Network (Basic Neural Network)
class ANN:

    """ Description -
            a. Initializes the random weights and biases in a manner to prevent vanishing and exploding gradients
            b. Find which derivative functions are to be used while back propagation based on the given activation function list
        Variables -
            a. weights = List containing numpy arrays for each layers except the input layer
            b. bias = List containing numpy arrays for each layers except the input layer
            c. neurons = List containing the number of neurons in each layer
            d. activationList = List containing the activation function to be used at each layer
            e. activationDashList = List containing the activation function derivative to be used at each layer
    """
    def __init__(self, neurons, activationList):
        # initialize the arrays for weights and biases
        self.weights = [np.random.randn(neurons[i - 1], neurons[i]) * np.sqrt(2 / neurons[i - 1]) for i in range(1, len(neurons))]
        self.bias = [np.random.randn(neurons[i]) * np.sqrt(2 / neurons[i - 1]) for i in range(1, len(neurons))]

        # store the list which has the activation function to be applied at the end of each layer
        self.neurons = neurons
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

    """ Description -
            a. Normalizes the input activations before feeding to the network
            b. Returns the output layer activation for all the input activation in the form of numpy array
        Variables -
            a. activation = Stores the current activation
            b. weightedSum = Stores the current weighted sum
    """
    def predict(self, x):
        # normalize the input activations
        activation = self.normalize(x)
        
        weightedSum = None
        # feed the input x through all the layers one by one
        for i in range(self.numOfLayers):
            weightedSum = np.add(np.matmul(activation, self.weights[i]), self.bias[i])
            activation = self.activationList[i](weightedSum)
        return activation
    
    """ Description -
            a. Normalizes the input activations before training
            b. Performs mini-batch gradient descent
            c. Performs dropout regularization on the input activations for each layer but the last
            d. Performs back propagation on each input batch
            e. Updates the weights and the biases of the model
            f. Plots the learning curve graph
        Variables -
            a. costList - Stores the cost for every iteration
            b. dw - List storing the derivative of cost function with respect to every weight in form of array
            c. db - List storing the derivative of cost function with respect to every bias in form of array
            d. dropoutMatrix - List storing the boolean arrays which denote which activations are to be dropped out for each layer
            e. keepProb - Probability for each neuron to exist after dropout
            f. learnRate - Factor with which the gradient is to be mulitplied which updating the weights and biases
            g. epochs - Number of passes over the entire dataset
            h. batchSize - Number of examples given at at time to perform gradient descent on
    """
    def train(self, x, y, learnRate, epochs, batchSize, keepProb):
        # store the cost at each iteration
        costList = []
        
        # normalize the input activations
        x = self.normalize(x)
        
        # repeat the optimization on entire dataset for epoch number of times
        for e in range(epochs):
            # feed the data in batches
            for b in range(0, len(x), batchSize):
                dw = [None] * self.numOfLayers
                db = [None] * self.numOfLayers
                weightedSum = [None] * (self.numOfLayers + 1)
                activation = [None] * (self.numOfLayers + 1)
                
                # get the dropout matrix
                dropoutMatrix = [np.random.rand(x[b : b + batchSize].shape[0], self.neurons[i]) < keepProb for i in range(self.numOfLayers)]

                # copy the input activations
                activation[0] = x[b : b + batchSize].copy()
                
                # feed the input activation in the network
                for i in range(self.numOfLayers):
                    weightedSum[i + 1] = np.add(np.matmul(dropoutMatrix[i] * activation[i], self.weights[i]), self.bias[i])
                    activation[i + 1] = self.activationList[i](weightedSum[i + 1])

                # calculate the cost
                cost = activation[self.numOfLayers] - y[b : b + batchSize]
                
                # print the cost and add the cost to costList
                temp_cost = ((cost ** 2) / batchSize).sum()
                costList.append(temp_cost)
                print("Epoch", e + 1, "|", "Batch", int(b / batchSize), "|", "Cost", temp_cost)

                # perform backpropagation
                i = self.numOfLayers
                while i > 0:
                    cost = cost * self.activationDashList[i - 1](weightedSum[i])
                    db[i - 1] = (cost / batchSize).sum(axis = 0)
                    dw[i - 1] = np.matmul((dropoutMatrix[i - 1] * activation[i - 1]).T, cost / batchSize)
                    cost = np.matmul(cost, self.weights[i - 1].T)
                    i -= 1

                # nudge the weights and biases
                for i in range(self.numOfLayers):
                    self.weights[i] -= learnRate * dw[i]
                    self.bias[i] -= learnRate * db[i]
                
        # plot the learning curve
        plt.plot(range(1, len(costList) + 1), costList)
        plt.scatter(range(1, len(costList) + 1), costList)        
        plt.show()
    
    """ Description -
            The following few functions return the non-linear/linear activation and their derivatives
    """
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
        return x * (x > 0)
    
    @staticmethod
    def reluDash(x):
        return x > 0
        
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
    
    """ Description -
            This function normalizes the input activations and returns the new array which is normalized and of same size
        Variable -
            a. mu - It stores the mean of all activations
            b. sigma_sq - It stores the variance of all activations
    """
    @staticmethod
    def normalize(x):
        temp = x.copy()
        mu = temp.sum(axis = 0) / float(len(temp))
        temp = temp - mu
        sigma_sq = (temp ** 2).sum(axis = 0) / float(len(temp))
        temp = temp * ((sigma_sq + 0.00000001) ** -0.5)
        return temp
   
    def evaluate(x, y):
        y_pred = self.predict(x)
        true_count = 0
        for i in range(len(y)):
            if np.argmax(y_pred[i]) == np.argmax(y[i]):
                true_count += 1
        print(true_count, "number of correct predictions of", len(y), "input samples")
