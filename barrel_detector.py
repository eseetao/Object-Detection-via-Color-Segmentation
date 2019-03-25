__author__ =  "Erik Seetao"

import os, cv2
from skimage.measure import label, regionprops
from skimage import morphology
from skimage import exposure

import pickle
import random
import numpy as np 
import skimage
import math

from skimage import color
#from Dataloader import DataLoader 
#from barrel_logistic_regression import logistic_regression

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class BarrelDetector():
	def __init__(self, window_len = 10, classes = 1, pickle_file = "/Users/eseetao/Documents/School Docs/ECE276A/Project 1/Blue_Barrel_Data.pickle"):
		'''
		Initilize your blue barrel detector with the attributes you need
		eg. parameters of your classifier

		Input:
			window_len: (int) the X by X window size, recommend setting to 10
			classes: (int) number of feature classes (1 for barrel, 2 for non-barrel blue and barrel)
			pickle_file: (str) directory path of pickled ROI files for barrel masks
		'''

		self.classes = 1
		self.window_len = 10

		#hardcoded training data, directly imported from logistic regression file
		self.weights = np.array([[0.7494962 ], [1.06116255], [1.08661419], [0.64272554], [0.9570562 ],
       [0.66388464], [0.75964282], [0.73766066], [0.91718386], [0.51647785],
       [0.36356528], [0.66398605], [0.63206386], [0.74923407], [0.69921265],
       [0.52133188], [1.05475385], [0.8150517 ], [1.17005364], [0.94020049],
       [0.62688683], [0.53138032], [0.49368532], [0.6336654 ], [0.92727291],
       [0.23127232], [0.69428842], [0.16409646], [0.9157514 ], [1.10575119],
       [0.24420043], [0.37388958], [0.50011574], [0.95303404], [0.61861707],
       [0.71310939], [0.72312144], [1.09122429], [0.17985324], [0.94488663],
       [0.70991088], [0.91065917], [1.01614244], [0.4631109 ], [0.92521567],
       [0.61209403], [0.77331126], [0.21868101], [0.99693909], [0.12463649],
       [0.84163063], [0.64775794], [1.01937042], [0.62873249], [0.89647378],
       [0.88911796], [0.18250419], [0.33782053], [0.82936287], [0.50110663],
       [0.71072024], [1.04006929], [0.1248057 ], [1.07545628], [0.51895396],
       [0.47063615], [0.96640687], [0.89320159], [0.20371353], [0.50399102],
       [0.38707094], [1.13868907], [1.09791764], [0.17275339], [0.74808489],
       [0.16713475], [1.0786482 ], [1.06672644], [0.85908534], [0.56848491],
       [0.60000734], [0.23559321], [0.90866115], [0.2284767 ], [0.4598201 ],
       [0.53181053], [1.05833759], [0.7164617 ], [0.12247307], [0.35926429],
       [0.25273107], [0.44431269], [0.8294086 ], [0.74324843], [0.50467524],
       [0.32294874], [0.98722572], [0.67354641], [1.10954157], [0.29070853]])

		self.bias = np.array([-7.19140105])




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
		sigmoid = 1/(1 + np.exp(-x)) #standard sigmoid (look at pg 12 from supervised learning slide) #DELETE THIS LATER
		return sigmoid #single class 

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

		epochs = 5
		learning_rate = 0.001
		epsilon = 0.00000001

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
				
				#perhaps standardize each feature vector (not normalize) to help with dark lighting

			print("Epoch number {0}: Cross entropy is {1}".format(epoch,cross_entropy[0,0])) #f-strings instead of other convention


		plt.plot(error_plot,label="Cross entropy")
		plt.xlabel("Number of Samples x100")
		plt.ylabel("Cross Entropy loss")
		plt.show()

	def test(self,image):
		'''
		Input:
			image: (np.array) np matrix of the test image
		Output:
			test_mask: (np.array) mask of the barrel on the test image
		'''

		test_mask = np.zeros((800, 1200, self.classes)) #for more general cases use .shape on image

		img_yuv = color.convert_colorspace(image,"RGB","YUV")[:,:,1] #index1 to grab "U" in YUV
		window_center = self.window_len // 2

		for m in range(window_center, 800 - (window_center)):
			for n in range(window_center, 1200 - (window_center)):

				window = img_yuv[m - (window_center): m + (window_center), n - (window_center): n + (window_center)]

				test_mask[m,n,:] = self.sigmoid(window.reshape(1,-1), self.weights, self.bias)

		return test_mask



	def segment_image(self, img):
		'''
		Calculate the segmented image using a classifier
		eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
		call other functions in this class if needed
			
		Inputs:
			img - original image
		Outputs:
			mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''

		img_predict = self.test(img)[:,:,0]

		img_range = np.max(img_predict) - np.min(img_predict)
		mask_img = (img_predict - np.min(img_predict))/img_range

		return mask_img

	def get_bounding_box(self, img):
		'''
		Find the bounding box of the blue barrel
		call other functions in this class if needed
			
		Inputs:
			img - original image
		Outputs:
			boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
			where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
			is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		boxes = []

		#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #uncomment to help plot

		grayscale_img = self.segment_image(img)
		thresh_mask = np.zeros((800,1200),dtype='uint8')
		thresh_mask[grayscale_img > 0.35] = 1 #seems like we need an adaptive thresholding
		
		contours, hierarchy = cv2.findContours(thresh_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		
		for contour in contours:

			if contour.shape[0] > 6: #originally 10
				x, y, width, height = cv2.boundingRect(contour)
				aspect_ratio = float(height)/width
				print("Aspect ratio is: ",aspect_ratio)

				(x_ell,y_ell), (minor_axis,major_axis), angle = cv2.fitEllipse(contour)
				print("Minor_axis: ",minor_axis, "and Major_axis is: ",major_axis)

				area = width*height
				print("Area is: ",area)

				if area > 600 and aspect_ratio > 1 and aspect_ratio < 4.2:

					boxes.append([x, y, (x + width), (y + height)]) #rc coordinates?

			else:
				print("Contour didn't pass, shape is less than 10")
		print(boxes)
		return boxes



if __name__ == '__main__':

	from Dataloader import DataLoader 
	from barrel_logistic_regression import logistic_regression

	training_directory = "/Users/eseetao/Documents/School Docs/ECE276A/Project 1/trainset"
	Data = DataLoader("/Users/eseetao/Documents/School Docs/ECE276A/Project 1/trainset",0.75) 

	predictor = BarrelDetector(10, 1, "/Users/eseetao/Documents/School Docs/ECE276A/Project 1/Blue_Barrel_Data.pickle")

	figure_num = 0
	for file_name in Data.training_set:
		figure_num += 1
		plt.figure(figure_num)
		file_name = "/Users/eseetao/Documents/School Docs/ECE276A/Project 1/trainset/" + file_name

		ax1 = plt.subplot(3,1,1)
		image = plt.imread(file_name)
		plt.imshow(image),plt.xticks([]),plt.yticks([]),plt.title("Original image")
		mask = predictor.segment_image(image)

		ax2 = plt.subplot(3,1,2)
		plt.imshow(mask,cmap="gray"),plt.xticks([]),plt.yticks([]),plt.title("Barrel Mask")
		
		
		boxes = predictor.get_bounding_box(image)
		ax3 = plt.subplot(3,1,3)
		plt.imshow(image),plt.xticks([]),plt.yticks([]),plt.title("Original image with bounding box")
		print(boxes)

		for box in boxes:
			bounding_box = patches.Rectangle((box[0],box[3]),box[2]-box[0],box[1]-box[3],linewidth=1,edgecolor='r',facecolor='none')
			ax3.add_patch(bounding_box)
		

		plt.show()

