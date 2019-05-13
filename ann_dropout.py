# import the required libraries
import numpy as np
import matplotlib.pyplot as plt

# define a class for the Artificial Neural Network (Basic Neural Network)
class ANN:


    def __init__(self, nn_dmn, act_funcs):
        """ 
        Description -
        a. Initializes the random weights and biases
        b. Find which derivative functions are to be used while back propagation based on the given activation function list
        Variables -
        a. weights - List containing numpy arrays for each layers except the input layer
        b. bias - List containing numpy arrays for each layers except the input layer
        c. nn_dmn - List containing the dimensions of each layer
        d. act_func - List containing the activation function to be used at each layer
        e. act_prime_funcs - List containing the activation function derivative to be used at each layer
        f. no_layers - Number of layers in the neural network
        """
        # initialize  weights and biases
        self.weights = [np.random.randn(nn_dmn[i - 1], nn_dmn[i]) * np.sqrt(2 / nn_dmn[i - 1]) for i in range(1, len(nn_dmn))]
        self.bias = [np.random.randn(nn_dmn[i]) * np.sqrt(2 / nn_dmn[i - 1]) for i in range(1, len(nn_dmn))]

        # store the activation functions list and the number of layers
        self.nn_dmn = nn_dmn
        self.act_funcs = act_funcs
        self.no_layers = len(act_funcs)

        # create a activation prime list
        self.act_prime_funcs = []
        for func in self.act_funcs:
            if func == self.sigmoid:
                self.act_prime_funcs.append(self.sigmoid_prime)
            elif func == self.relu:
                self.act_prime_funcs.append(self.relu_prime)
            elif func == self.tanh:
                self.act_prime_funcs.append(self.tanh_prime)
            elif func == self.linear:
                self.act_prime_funcs.append(self.linear_prime)

    def forward_prop(self, x):
        """
        Description -
        a. Normalizes the input activations before feeding to the network
        b. Returns the output layer activation for all the input activation in the form of numpy array
        Variables -
        a. act = Stores the current activation
        b. wt_sum = Stores the current weighted sum
        """
        # normalize the input activations
        act = self.normalize(x)
        wt_sum = None
        
        # feed the input x through all the layers one by one
        for i in range(self.no_layers):
            wt_sum = act @ self.weights[i] + self.bias[i]
            act = self.act_funcs[i](wt_sum)
            
        return act

    def forward_prop_mem(self, x, drop_mat):
        """
        Description -
        a. Normalizes the input activations before feeding to the network
        b. Returns the output layer activation for all the input activation in the form of numpy array
        c. Also returns the activations and the weighted sum of all the layers
        Variables -
        a. act = Stores the current activation
        b. wt_sum = Stores the current weighted sum
        """
        wt_sum = [None] * (self.no_layers + 1)
        act = [None] * (self.no_layers + 1)
        
        # normalize the input activations
        act[0] = self.normalize(x)
        
        # feed the input x through all the layers one by one
        for i in range(self.no_layers):
            wt_sum[i + 1] = (drop_mat[i] * act[i]) @ self.weights[i] + self.bias[i]
            act[i + 1] = self.act_funcs[i](wt_sum[i + 1])
            
        return wt_sum, act

    def back_prop(self, delta, wt_sum, act, drop_mat, batch_size, cost_func):
        """
        Description -
        Finds the derivative of the cost with respect to the weights and biases
        Variables -
        a. dw - List storing the derivative of cost function with respect to every weight in form of array
        b. db - List storing the derivative of cost function with respect to every bias in form of array
        """
        # initialize the derivative terms
        dw = [None] * self.no_layers
        db = [None] * self.no_layers
        
        i = self.no_layers
        while i > 0:
            # dont mutiply the activation prime term in case of cross entropy cost function
            if cost_func != 'cross_entropy' or i != self.no_layers:
                delta = delta * self.act_prime_funcs[i - 1](wt_sum[i])
            # find the derivative wrt biases
            db[i - 1] = delta.sum(axis = 0) / batch_size
            # find the derivative wrt weights
            dw[i - 1] = ((drop_mat[i - 1] * act[i - 1]).T @ delta) / batch_size
            # update delta
            delta = np.matmul(delta, self.weights[i - 1].T)
            i -= 1
            
        return dw, db

    def update(self, dw, db, learn_rate):
        """
        Description -
        This function updates the weights and biases list according to the input learning rate and the derivatives
        """
        for i in range(self.no_layers):
            self.weights[i] -= learn_rate * dw[i]
            self.bias[i] -= learn_rate * db[i]
            
    def train(self, x, y, learn_rate, epochs, batch_size, keep_prob, cost_func = 'mean_square'):
        """ 
        Description -
        a. Normalizes the input activations before training
        b. Performs mini-batch gradient descent
        c. Performs dropout regularization on the input activations for each layer but the last
        d. Performs back propagation on each input batch
        e. Updates the weights and the biases of the model
        f. Plots the learning curve graph
        Variables -
        a. cost_at_itr - Stores the cost for every iteration
        b. dw - List storing the derivative of cost function with respect to every weight in form of array
        c. db - List storing the derivative of cost function with respect to every bias in form of array
        d. drop_mat - List storing the boolean arrays which denote which activations are to be dropped out for each layer
        e. keep_prob - Probability for each neuron to exist after dropout
        f. learn_rate - Factor with which the gradient is to be mulitplied which updating the weights and biases
        g. epochs - Number of passes over the entire dataset
        h. batch_size - Number of examples given at at time to perform gradient descent on
        i. cost_func - Defines the type of cost function used
        """
        # store the cost at each iteration
        cost_at_itr = []
        
        # normalize the input activations
        x = self.normalize(x)
        
        # repeat optimization epoch times
        for e in range(epochs):
            # feed the data in batches
            for b in range(0, len(x), batch_size):
                # define the dropout matrix
                drop_mat = [np.random.rand(x[b : b + batch_size].shape[0], self.nn_dmn[i]) < keep_prob for i in range(self.no_layers)]

                # perform foward propagation for the current batch
                wt_sum, act = self.forward_prop_mem(x[b : b + batch_size], drop_mat)
                
                # calculate the delta for the output layer
                delta = act[self.no_layers] - y[b : b + batch_size]

                # print the cost and add the cost to cost_at_itr list
                cost = (delta**2 / batch_size).sum()
                
                cost_at_itr.append(cost)
                print("Epoch", e + 1, "|", "Batch", int(b / batch_size), "|", "Cost", cost)
                
                # perform backpropagation
                dw, db = self.back_prop(delta, wt_sum, act, drop_mat, batch_size, cost_func)

                # update the weights and biases
                self.update(dw, db, learn_rate)
                
        # plot the learning curve
        plt.plot(range(1, len(cost_at_itr) + 1), cost_at_itr)
        plt.scatter(range(1, len(cost_at_itr) + 1), cost_at_itr)        
        plt.show()

    @staticmethod
    def normalize(x):
        """ 
        Description -
        This function normalizes the input activations and returns the new array which is normalized and of same size
        Variable -
            a. mean - It stores the mean of all activations
            b. var - It stores the variance of all activations
        """
        temp = x.copy()
        mean = temp.sum(axis = 0) / float(len(temp))
        temp = temp - mean
        var = (temp ** 2).sum(axis = 0) / float(len(temp))
        temp = temp / ((var ** 0.5) + 0.00000001)
        return temp
    
    """ Description -
            The following few functions return the non-linear/linear activation and their derivatives
    """
    # define the activation functions and the derivative functions
    @staticmethod
    def sigmoid(x):
        temp = 1.0 / (1.0 + np.exp(-x))
        return temp

    @staticmethod
    def sigmoid_prime(x):
        temp = (1.0 / (1.0 + np.exp(-x))) * (1.0 - (1.0 / (1.0 + np.exp(-x))))
        return temp
    
    @staticmethod
    def relu(x):
        return x * (x > 0)
    
    @staticmethod
    def relu_prime(x):
        return x > 0
        
    @staticmethod
    def tanh(x):
        temp = np.tanh(x)
        return temp

    @staticmethod
    def tanh_prime(x):
        temp = 1.0 - tanh(x) ** 2
        return temp

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_prime(x):
        return 1

    def evaluate(self, x, y):
        """
        Description -
        This function returns the correctly predicted samples of the given features set by comparing with the target set
        Variable -
        y_pred - Stores the output layer activations got from the neural network
        true_count - Stores the correctly predicted samples count
        """
        y_pred = self.forward_prop(x)
        true_count = 0
        for i in range(len(y)):
            if np.argmax(y_pred[i]) == np.argmax(y[i]):
                true_count += 1
        print(true_count, "/", len(y))
