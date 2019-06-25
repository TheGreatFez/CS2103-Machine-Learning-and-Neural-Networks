# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 11:16:00 2019

@author: Adi
"""
import numpy as np
from PerceptronClass import Perceptron
#Debugging Functions
def printArray(x):
    for i in range(len(x)):
        for a in range(len(x[i])):
            print("Row " + str(i),end = " ")
            print("Column " + str(a),end = " ")
            print(x[i][a])


class NeuralNetwork:
    def __init__(self,network_setup):
        
        self.weight_max = 1
        self.weight_min = -1
        '''
        sets the limits of the weights
        '''
        
        self.generation = 0
        '''
        generation is the number of iterations of training the neural network
        has undergone
        '''
        self.net_setup = network_setup
        '''
        net_setup is a list of values. Each value is how many perceptrons are 
            on each levels
            a preceptron is a node that takes in inputs from the previous level
            net_setup.length = number of inner network levels, including the input and output level
            ie net_setup = [2,3,2]: 2 inputs, 3 preceptrons, 2 outputs, total of 3 levels
            
        net_setup must have a length of at least 2
        '''
        self.network = []
        #^^^holds a 2D array with perceptrons organized like net_setup dictates        
        for size in self.net_setup:
            level = np.zeros((size,1))
            self.network.append(level)
        
        self.weights = []
        count = 0
        for net_index in range(len(self.net_setup[1:])):
            diff = self.weight_max - self.weight_min
            temp_weight_array = diff*np.random.rand(self.net_setup[net_index + 1],self.net_setup[net_index]) + self.weight_min
            self.weights.append(temp_weight_array)
            print('Weights of Layer ' + str(count))
            print(temp_weight_array)
            count += 1
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def getOutput(self,input_data):
        output = 0
        count = 0
        prev_level = input_data
        self.network[0] = prev_level
        for weights_array in self.weights:
            next_level = np.dot(weights_array,prev_level)
            for i in range(len(next_level)):
                next_level[i] = self.sigmoid(next_level[i])
            print('Count ' + str(count))
            print(next_level)
            prev_level = next_level
            self.network[count + 1] = prev_level
            count += 1
#        temp_array = []
        
        return output
    
data = [{'data': [3,1.5], 'target': 1}, {'data':[2,1], 'target': 0},
        {'data': [4,1.5], 'target': 1}, {'data': [3,1], 'target': 0},
        {'data': [3.5,.5], 'target': 1}, {'data': [2,.5], 'target': 0},
        {'data': [5.5,1], 'target': 1}, {'data': [1,1], 'target': 0}]
n = NeuralNetwork([2,3,3,1])

print(n.getOutput(data[0]['data']))

print('Network Output')
print(n.network)