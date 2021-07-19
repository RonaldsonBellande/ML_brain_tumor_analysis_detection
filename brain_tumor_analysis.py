from __future__ import print_function
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sys 
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, cross_val_predict, StratifiedKFold, train_test_split, learning_curve, ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV,KFold, cross_val_predict, StratifiedKFold, train_test_split, learning_curve, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.wrappers.scikit_learn import KerasClassifier
import keras
from keras.datasets import cifar10
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras import layers
import glob
import shutil
import tensorflow as tf
import matplotlib.image as img
import os, os.path
import shutil
# from IPython import get_ipython
from multiprocessing import Pool
warnings.filterwarnings('ignore')
# get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')


class utilities(object):

    def __init__(self, name_of_new_directory = "brain_cancer_seperate/"):
        self.path = "/Data2"
        self.seperate_path = "Data2/Data"
        self.file_path_to_move = "brain_cancer_seperate/"
        self.valid_images = [".jpg",".png"]
        self.name_of_new_directory = name_of_new_directory



    # To create more nested folders you need to seperate he awway with - and that teels that it is a seperate folder
    def seperate_image_base_on_image(self, nested_folders = "None", directory_name = "True - False"):
        
        directory_array = directory_name.split(" - ")
        if nested_folders == "None":
            if os.path.isdir(self.name_of_new_directory) == False:
                os.mkdir(self.name_of_new_directory)
        else:
            for i in range(len(directory_array)):
                if os.path.isdir(str(self.name_of_new_directory + directory_array[i])) == False:
                    os.mkdir(str(self.name_of_new_directory + directory_array[i]))


    # To seperate images base on image name
    def seperate_image_into_file(self):
        list_images = os.listdir(self.seperate_path)
        for image in list_images:
            if image.endswith(self.valid_images[0]) or image.endswith(self.valid_images[1]):
                if 'y' in image.lower():
                    shutil.copy(os.path.join(self.seperate_path, image), self.file_path_to_move + "True")
                elif 'n' in image.lower():
                    shutil.copy(os.path.join(self.seperate_path, image), self.file_path_to_move + "False")
                else:
                    print("error")




class brain_cancer_analysis(object):

    def __init__(self, number_classes = 2):    
        self.images = []
        self.filename = []
        self.image_file = []
        # 0 for False and 1 for True for label name
        self.label_name = []
        self.number_classes = number_classes
        self.image_size = 240
        self.path = "Data/"
        self.true_path  = "brain_cancer_seperate/"
        self.valid_images = [".jpg",".png"]
        self.categories = ["False","True"]
        self.input_shape = None
        self.advanced_categories = ["False", "True", "Degree1", "Degree2"]

        # Split training data variables
        self.X_train = None
        self.X_test = None
        self.Y_train_vec = None
        self.Y_test_vec = None

        # model informations
        self.model = None

        self.batch_size = [10, 20, 40, 60, 80, 100]
        self.epochs = [10, 50, 100]
        self.param_grid = dict(batch_size = self.batch_size, epochs = self.epochs)
        self.callbacks = keras.callbacks.EarlyStopping(monitor='val_acc', patience=4, verbose=1)


        
        # Brain Cancer true or false
        if self.number_classes == 2:
            # Check validity
            self.check_valid(self.categories[0])
            self.check_valid(self.categories[1])
            
            # resize image
            self.resize_image_and_label_image(self.categories[0])
            self.resize_image_and_label_image(self.categories[1])

        # Brain Cancer true, false, Degree1, Degree2
        elif self.number_classes == 4:
            # Check validity
            self.check_valid(self.advanced_categories[0])
            self.check_valid(self.advanced_categories[1])
            self.check_valid(self.advanced_categories[2])
            self.check_valid(self.advanced_categories[3])
            
            # Resize image
            self.resize_image_and_label_image(self.advanced_categories[0])
            self.resize_image_and_label_image(self.advanced_categories[1])
            self.resize_image_and_label_image(self.advanced_categories[2])
            self.resize_image_and_label_image(self.advanced_categories[3])

        else:
            print("Detection Variety out of bounds")

        
        # Numpy array
        self.image_file = np.array(self.image_file)
        self.label_name = np.array(self.label_name)
        self.label_name = self.label_name.reshape((len(self.image_file),1))

        self.splitting_data_normalize()
        self.create_models_1()


    # Checks to see if the image is valid or not
    def check_valid(self, input_file):
        for img in os.listdir(self.true_path + input_file):
            ext = os.path.splitext(img)[1]
            if ext.lower() not in self.valid_images:
                continue
    

    # Resize images
    def resize_image_and_label_image(self, input_file):
        for image in os.listdir(self.true_path + input_file):
            
            image_resized = cv2.imread(os.path.join(self.true_path + input_file,image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            self.image_file.append(image_resized)

            if input_file == "False":
                self.label_name.append(0)
            elif input_file == "True":
                self.label_name.append(1)
            elif input_file == "Degree1":
                self.label_name.append(2)
            elif input_file == "Degree2":
                self.label_name.append(3)
            else:
                print("error")


    # Split training data and testing Data and makes it random and normalized it
    def splitting_data_normalize(self):
        self.X_train, self.X_test, self.Y_train_vec, self.Y_test_vec = train_test_split(self.image_file, self.label_name, test_size = 0.15, random_state = 42)

        self.input_shape = self.X_train.shape[1:]
        
        self.Y_train = keras.utils.to_categorical(self.Y_train_vec, self.number_classes)
        self.Y_test = keras.utils.to_categorical(self.Y_test_vec, self.number_classes)

        # Normalize
        self.X_train = self.X_train.astype("float32")
        self.X_train /= 255
        self.X_test = self.X_test.astype("float32")
        self.X_test /= 255


    def create_models_1(self):
        
        self.model = Sequential()
        # First Hitten Layer with 64, 7, 7
        self.model.add(Conv2D(64,(7,7), strides = (1,1), padding="same", input_shape = self.input_shape, activation = "relu"))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size = (4,4)))
        self.model.add(Dropout(0.25))
    
        # Second Hitten Layer 32, 7, 7
        self.model.add(Conv2D(32,(7,7), strides = (1,1), padding="same", activation = "relu"))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(Dropout(0.25))
    
        # Third Hitten Layer 32, 7, 7
        # self.model.add(Conv2D(16,(7,7), strides = (1,1), padding="same", activation = "relu"))
        # self.model.add(MaxPooling2D(pool_size = (1,1)))
        # self.model.add(Dropout(0.25))
    
        # last layer, output Layer
        self.model.add(Flatten())
        self.model.add(Dense(units = self.number_classes, activation = 'softmax', input_dim=2))

        self.model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    
    def create_models_2(self):
        self.model = Sequential()
        self.model.add(Conv2D(64, (3,3), input_shape = X_train.shape[1:]))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size = (2,2))) # Pooling

        self.model.add(Conv2D(64, (3,3), input_shape = X_train.shape[1:]))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size = (2,2))) # Pooling


        self.model.add(Flatten())
        self.model.add(Dense(64))

        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    #  Training model 
    def train_model(self):
       
        self.model = KerasClassifier(build_fn = ConvNeuralNetwork, verbose = 0)
        grid = GridSearchCV(estimator = self.model, param_grid = self.param_grid, n_jobs = 1, cv = 3, verbose = 10)

        self.brain_cancer_model = model.fit(self.X_train, self.Y_train,
          batch_size=self.batch_size[2],
          validation_split=0.15,
          epochs=self.epochs[2],
          callbacks=[self.callbacks],
          shuffle=True)
    
    # Evaluate model
    def evaluate_model(self):
        evaluation = model.evaluate(X_test, Y_test, verbose=1)
        print("Loss:", evaluation[0])
        print("Accuracy: ", evaluation[1])


    # PLotting model
    def plot_model(brain_cancer_model):
        plt.plot(brain_cancer_model.history['accuracy'])
        plt.plot(brain_cancer_model.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.show()
        plt.plot(brain_cancer_model.history['loss'])
        plt.plot(brain_cancer_model.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.show()


if __name__ == "__main__":
    
    # Determin if you want to create new files, second augument is where you create a new folder
    util = utilities()
    if len(sys.argv) != 1:
        if sys.argv[1] == "create":
            util.seperate_image_base_on_image(nested_folders = "True")

        # Seperate images base on names
        if sys.argv[1] == "seperate":
            util.seperate_image_into_file()


    # Begin analysis
    brain_analysis = brain_cancer_analysis()






       








    
