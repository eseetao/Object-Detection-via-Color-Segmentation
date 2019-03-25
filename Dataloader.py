__author__ = "Erik Seetao"

import os
import pickle
import random
import numpy as np 
import skimage
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from skimage import color
from roipoly import RoiPoly
from pathlib import Path

#print(matplotlib.__version__) #current version is 2.2.2
#print(np.__version__) #current version 1.14.5
#print(cv2.__version__) #current version 4.0.0

class DataLoader:

    def __init__(self,source,training_split):

        '''
        Input:
            training_split: (float) fraction you want to split into training, the rest will be put into validation
        '''

        barrel_images = os.listdir(source)
        training_size = int(len(barrel_images)*training_split)

        self.mask = np.ones((800,1200))

        self.training_set = barrel_images[:training_size]
        self.validation_set = barrel_images[training_size:]

        self.file_numbers = len(barrel_images)
        self.root_dir = "/Users/eseetao/Documents/School Docs/ECE276A/Project 1/trainset/"
        #self.root_dir = "/Users/eseetao/Documents/School Docs/ECE276A/Project 1/testpickle/" #on a test pickle 

        #print(self.training_set)
        #print(self.validation_set)


    def label_images(self,pickle_file):
        '''
        Use ROI poly to create binary masks of barrels and load into a pickle file
        Input:
            pickle_file: (str) directory path for pickle_file
        Output:
            pickle file of ROI barrel masks
        '''
        
        #folder = "trainset"
        #folder_path = "/Users/eseetao/Documents/School Docs/ECE276A/Project 1/trainset"

        stored_data = [] #storing image and mask 
        counter = 0 #for 
        total_files = self.training_set+self.validation_set

        for filename in os.listdir(self.root_dir):
            #find masks on all the images, not just the training/validation sets separately

            print(filename)
            data = {"image":plt.imread(self.root_dir + total_files[counter])} #create dictionary data
            
            #show first image
            img = plt.imread(self.root_dir + filename) #load image
            fig = plt.figure()
            plt.imshow(img, interpolation='nearest', cmap="Greys")
            plt.colorbar()
            plt.title("left click: line segment         right click: close region")
            plt.show(block=False)

            #populate data with RGB values
            #data["R"] = img[:,:,0]
            #data["G"] = img[:,:,1]
            #data["B"] = img[:,:,2]
            #RGB not needed, do YUV instead

            YUV = color.convert_colorspace(data["image"],"RGB","YUV") #YUV color scale
            data["Y"] = YUV[:,:,0]
            data["U"] = YUV[:,:,1]
            data["V"] = YUV[:,:,2]

            #print("YUV values are ",data["Y"],data["U"],data["V"]) 

            img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114]) #convert to grayscale to impose mask on
            
            #user draws first roi
            roi1 = RoiPoly(color='r', fig=fig)

            #show image with first roi
            fig = plt.figure()
            plt.imshow(img, interpolation='nearest', cmap="Greys")
            plt.colorbar()
            roi1.display_roi() #can take out


            decision = input("Label another ROI? Y/N")
            if decision == 'Y':
                # Let user draw second ROI
                roi2 = RoiPoly(color='b', fig=fig)

                data["mask_barrel"] = roi1.get_mask(img_gray) + roi2.get_mask(img_gray)
                plt.imshow(data["mask_barrel"],interpolation='nearest', cmap="gray")
                plt.title('ROI masks of the two ROIs')
                plt.show()

            else:
                # Show single ROI mask
                data["mask_barrel"] = roi1.get_mask(img_gray)
                plt.imshow(data["mask_barrel"],interpolation='nearest', cmap="gray")
                plt.title('ROI masks of the ROI')
                plt.show()

            counter += 1
            stored_data.append(data)


        with open(pickle_file, 'wb') as pickle_handle:
            pickle.dump(stored_data, pickle_handle, protocol=pickle.HIGHEST_PROTOCOL)

    def unpickle_data(self,pickle_file):
        '''
        Unpickle the mask pickle file and yield the binary 0 or 1 if a window belongs to a barrel class
        
        Input:
            pickle_file: (str) directory path for pickled file
        Output:
            (np.array) yielded window with binary classification
        '''

        print("Unpickling data")

        window_len = 10 #recommend length of 10
        slide_size = 5  #recommend half of window length

        with open(pickle_file, 'rb') as pickle_handle:
            stored_data = pickle.load(pickle_handle) 

        training_length = len(self.training_set)
        window_center = window_len // 2 

        for data in stored_data[:training_length]:

            mask = data["mask_barrel"]
            img_blueness = data["U"] #recommend only looking at U value to separate "blueness"
            
            for m in range(window_center, img_blueness.shape[0] - window_center, slide_size):
                for n in range(window_center, img_blueness.shape[1] - window_center, slide_size):
                    #use generator
                    
                    value = img_blueness[m - window_center: m + window_center, n - window_center:n + window_center], mask[m,n]
                    #print(value) #damage control
                    yield value
            

if __name__ == "__main__":

    print(os.getcwd())

    #for one class
    Data = DataLoader("/Users/eseetao/Documents/School Docs/ECE276A/Project 1/trainset",0.8)
    Data.label_images("/Users/eseetao/Documents/School Docs/ECE276A/Project 1/Blue_Barrel_Data.pickle")

    #multiclass (blue barrel, blue non barrel, else)
    #Data = DataLoader("/Users/eseetao/Documents/School Docs/ECE276A/Project 1/trainset",0.8)
    #Data.label_images("/Users/eseetao/Documents/School Docs/ECE276A/Project 1/Blue_Barrel_Data_multiclass.pickle")


