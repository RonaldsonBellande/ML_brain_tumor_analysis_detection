from header_imports import *


class computer_vision_transfer_learning(object):
    def __init__(self, currently_build_model, image_type, category):
        
        self.image_file = []
        self.label_name = []
        self.image_size = 224
        self.path  = "traffic_sign:s/"
        self.image_type = image_type
        self.category = category
        self.valid_images = [".jpg",".png"]
        self.model_summary = "model_summary/"
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

        self.path_to_model = "models/"
        self.save_model_path = self.path_to_model + "/transfer_learning_model/"
        self.currently_build_model = self.path_to_model + currently_build_model
        self.model.load_weights(self.currently_build_model)
        self.transfer_learning_model()
        self.setup_structure()
        self.save_model_summary()


    def setup_structure(self):

        if self.image_type == "small_traffic_sign":
            self.true_path = self.path + "Small_Traffic_Sign/"
        elif self.image_type == "regular":
            self.true_path = self.path + "Train/"
        elif self.image_type == "train1":
            self.true_path = self.path + "Train_1_50/"
        elif self.image_type == "train2":
            self.true_path = self.path + "Train_2_25/"
        elif self.image_type == "train3":
            self.true_path = self.path + "Train_3_25/"

        self.advanced_categories = ["0", "1", "2", "2", "3", "4", "5", "6", "7", "8", "9", "10","11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30","31", "32", "33", "34", "35", "36", "37", "38","39", "40", "41", "42"]
        self.categories = ["One Way Right", "Slow Xing", "Yield", "One Way Left", "Traffic Light Sign", "Stop", "Ducky"]
        self.category_names = traffic_sign_categories.category_names

        if self.category == "normal":
            self.model_categories = self.categories
            self.number_classes = 7
        elif self.category == "regular":
            self.model_categories = self.category_names
            self.number_classes = 43

        if self.category == "regular":
            for i in range(0, 43):
                self.check_valid(self.advanced_categories[i])
        elif self.category == "normal":
            for i in range(0, 7):
                self.check_valid(self.categories[i])

        if self.category == "regular":
            for i in range(0,43):
                self.resize_image_and_label_image(self.advanced_categories[i])
        elif self.category == "normal":
            for i in range(0,7):
                self.resize_image_and_label_image(self.categories[i])


    def resize_image_and_label_image(self, input_file):

        for image in os.listdir(self.true_path + input_file):    
            image_resized = cv2.imread(os.path.join(self.true_path + input_file,image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            self.image_file.append(image_resized)

            if self.category == "regular":
                for i in range(0, 43):
                    print(i)
                    if input_file == str(i):
                        self.label_name.append(i)
            
            elif self.category == "normal":
                if input_file == "One Way Right":
                    self.label_name.append(0)
                elif input_file == "Slow Xing":
                    self.label_name.append(1)
                elif input_file == "Yield":
                    self.label_name.append(2)
                elif input_file == "One Way Left":
                    self.label_name.append(3)
                elif input_file == "Traffic Light Sign":
                    self.label_name.append(4)
                elif input_file == "Stop":
                    self.label_name.append(5)
                elif input_file == "Ducky":
                    self.label_name.append(6)
                else:
                    print("error")

        self.image_file = np.array(self.image_file)
        self.label_name = np.array(self.label_name)
        self.label_name = self.label_name.reshape((len(self.image_file),1))


    def check_valid(self, input_file):
        for img in os.listdir(self.true_path + input_file):
            ext = os.path.splitext(img)[1]
            if ext.lower() not in self.valid_images:
                continue


    def create_models_1(self):

        self.model = Sequential()
        self.model.add(Conv2D(filters=64, kernel_size=(7,7), strides = (1,1), padding="same", input_shape = self.input_shape, activation = "relu"))
        self.model.add(MaxPooling2D(pool_size = (4,4)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(filters=32, kernel_size=(7,7), strides = (1,1), padding="same", activation = "relu"))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(filters=16, kernel_size=(7,7), strides = (1,1), padding="same", activation = "relu"))
        self.model.add(MaxPooling2D(pool_size = (1,1)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(units = self.number_classes, activation = "softmax", input_dim=2))
        self.model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        return self.model

    
    def create_models_2(self):

        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape = self.input_shape))
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Flatten())
        self.model.add(Dense(units=self.number_of_nodes, activation="relu"))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units = self.number_classes, activation="softmax"))
        self.model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics= ['accuracy'])
	
        return self.model


    def create_model_3(self):

        self.model = Sequential()
        self.MyConv(first = True)
        self.MyConv()
        self.MyConv()
        self.MyConv()
        self.model.add(Flatten())
        self.model.add(Dense(units = self.number_classes, activation = "softmax", input_dim=2))
        self.model.compile(loss = "binary_crossentropy", optimizer ="adam", metrics= ["accuracy"])
        
        return self.model
        

    def MyConv(self, first = False):
        if first == False:
            self.model.add(Conv2D(64, (4, 4),strides = (1,1), padding="same",
                input_shape = self.input_shape))
        else:
            self.model.add(Conv2D(64, (4, 4),strides = (1,1), padding="same",
                 input_shape = self.input_shape))
    
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(32, (4, 4),strides = (1,1),padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.25))


    def create_model_4(self):
        
        for layer in self.model.layers:
            layer.trainable = False
    
        self.model.add(Flatten())
        self.model.add(Dense(units = self.number_classes, activation = "softmax"))
        self.model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        return self.model


    def save_model_summary(self):
        with open(self.model_summary + self.create_model_type +"_summary_architecture_transfer_learning_" + str(self.number_classes) +".txt", "w+") as model:
            with redirect_stdout(model):
                self.model.summary()
