__author__ = "Erik Seetao"
__PID__ = "A10705834"

#import os
import pickle
import random
import numpy as np 
import skimage
import cv2
import pickle
import Dataloader

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from skimage import color
from roipoly import RoiPoly
from pathlib import Path
from Dataloader import DataLoader 

from skimage.morphology import disk,square,erosion,dilation
from skimage import exposure,color
import cv2

class logistic_regression:
    def __init__(self,window_len,classes):

        '''
	Input:
		window_len: (int) the X by X window size, recommend setting to 10
		classes: (int) number of feature classes (1 for barrel, 2 for non-barrel blue and barrel)
        '''

        self.root_dir = "/Users/eseetao/Documents/School Docs/ECE276A/Project 1/"

        #test overwrite user input
        self.window_len = 10 #recommend window length of 10, overwrite for now
        self.classes = 1

        #initialize random weight and bias
        self.weights = np.random.rand(self.window_len**2,self.classes) #want a len(weights)=w^2 since unpacking and appending each row
        self.bias = np.random.rand(1,self.classes)

    @staticmethod
    def sigmoid(data,weights,bias):
        '''
        for single class (barrel) logistic regression
	Input:
		data: (np.array) features for barrel class
		weights: (np.array) vector of weights corresponding to each feature
		bias: (float) binary 0 or 1 classification
	Output:
		(np.array) normalized softmax for single clas
	'''

        x = np.dot(data,weights) + bias
        sigmoid = 1/(1 + np.exp(-x)) 
        return sigmoid 

    @staticmethod
    def softmax(data,weights,bias):
        '''
        for multiclass logistic regression
	Input:
		data: (np.array) features for multiclass
		weights: (np.array) vector of weights corresponding to each feature
		bias: (float) bias during training
	Output:
		(np.array) normalized softmax for multiclass
	'''

        x = np.dot(data,weights) + bias
        x = np.exp(x - np.max(x))
        x = x/np.sum(x,1)
        return x

    def train(self,barrel_list,epochs,learning_rate,epsilon):
        '''
	logistic regression training

	Input:
		barrel_list: (np.array) features and label from unpickled file
		epochs: (int) learning epochs
		learning_rate: (float) learning rate
		epsilon: (float) logistic regression error
	Output:
		The updated self.weights and self.bias for test function calling sigmoid/softmax
	'''

        sample_counter = 0
        error_plot = []
        for epoch in range(epochs):
            for feature,label in barrel_list:

                y = self.sigmoid(feature.reshape(1,-1),self.weights,self.bias) 
                grad = label - y

                self.weights = self.weights + learning_rate * grad * feature.reshape(-1,1)
                self.bias = self.bias + learning_rate * grad
                sample_counter += 1

                if sample_counter % 100 == 0: #plot per hundred
                    cross_entropy = -1 * label * np.log(y + epsilon) - (1 - label) * np.log(1 - y + epsilon)
                    error_plot.append(cross_entropy[0,0])
                
                #standardize each feature vector (not normalize) to help with dark lighting

            print("Epoch number {0}: Cross entropy is {1}".format(epoch,cross_entropy[0,0])) #use f-strings
        

        plt.plot(error_plot,label="Cross entropy")
        plt.xlabel("Number of Samples x100")
        plt.ylabel("Cross Entropy loss")
        plt.show()

        #For user's debugging
        # print("Printing weight matrix: ")
        # print(list(self.weights.reshape(1, -1)))
        # print("Finished printing weight matrix")

        # print("Printing bias: ")
        # print(self.bias)
        # print("Finished printing bias")

    def test(self,image):
        '''
	Input:
		image: (np.array) np matrix of the test image
	Output:
		test_mask: (np.array) mask of the barrel on the test image
	'''

        test_mask = np.zeros((800, 1200, self.classes)) #for more general cases use .shape on image

        image = color.convert_colorspace(image,"RGB","YUV")[:,:,1] #index1 to grab "U" in YUV
        window_center = self.window_len // 2

        for m in range(window_center, 800 - (window_center)):
            for n in range(window_center, 1200 - (window_center)):

                window = image[m - (window_center): m + (window_center), n - (window_center): n + (window_center)]
                test_mask[m,n,:] = self.sigmoid(window.reshape(1,-1), self.weights, self.bias)

        return test_mask


if __name__ == "__main__":

    training_images = DataLoader("/Users/eseetao/Documents/School Docs/ECE276A/Project 1/trainset",0.90) #TA recommends 75%. try 90 
    Data = training_images.unpickle_data("/Users/eseetao/Documents/School Docs/ECE276A/Project 1/Blue_Barrel_Data.pickle") #currently window_len = 20, slide_size = 10
    #the unpickled data handles training set

    log_regression_model = logistic_regression(10,1)
    log_regression_model.train(Data,1000,0.01,0.00000001)

    with open("/Users/eseetao/Documents/School Docs/ECE276A/Project 1/model1.pickle", 'wb') as pickle_handle:
        pickle.dump(log_regression_model, pickle_handle, protocol=pickle.HIGHEST_PROTOCOL)

    #print(log_regression_model)
    
    figure_num = 0
    for file_name in training_images.training_set:
        figure_num = figure_num + 1
        plt.figure(figure_num)
        file_name = training_images.root_location + file_name
        print(file_name)
        file_name = "/Users/eseetao/Documents/School Docs/ECE276A/Project 1/trainset/" + file_name

        plt.subplot(2,1,1)
        image = plt.imread(file_name)
        plt.imshow(image),plt.xticks([]),plt.yticks([]),plt.title("Original image")
        mask = log_regression_model.test(image)
        plt.subplot(2,1,2)
        mask = mask[:,:,0]>0.7 #recommended threshold

        #use morphology from skimage
        #radius 10 disk, returns binary structured element
        mask = dilation(mask, disk(10))
        plt.imshow(mask,cmap="gray"),plt.xticks([]),plt.yticks([]),plt.title("Blue Barrel Mask")
        plt.savefig("/Users/eseetao/Documents/School Docs/ECE276A/Project 1/results10x10/figure{}.png".format(figure_num)) #one class (barrel blueness)
        #plt.show() 
    
