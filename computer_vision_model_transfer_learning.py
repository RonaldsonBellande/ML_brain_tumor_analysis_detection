from header_imports import *


class computer_vision_transfer_learning(object):
    def __init__(self, currently_build_model, image_type, category):
        
        self.image_file = []
        self.label_name = []
        self.number_classes = int(number_classes)
        self.image_size = 240
        self.number_of_nodes = 16
        self.true_path  = "brain_cancer_category_2/"
        self.image_type = image_type

        self.valid_images = [".jpg",".png"]
        self.categories = ["False","True"]
        self.advanced_categories = ["False", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
        self.model = None
        self.model_summary = "model_summary/"
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.model_type = model_type
        
        self.setup_structure() 
        self.splitting_data_normalize()
        
        if self.model_type == "model1":
            self.create_models_1()
        elif self.model_type == "model2":
            self.create_models_2()
        elif self.model_type == "model3":
            self.create_model_3()

        self.batch_size = [10, 20, 40, 60, 80, 100]
        self.epochs = [1, 5, 10, 50, 100, 200]
        self.number_images_to_plot = 16
        self.graph_path = "graph_charts/"
        self.model_path = "models/" 
        self.param_grid = dict(batch_size = self.batch_size, epochs = self.epochs)
        self.callback_1 = TensorBoard(log_dir="logs/{}-{}".format(self.model_type, int(time.time())))
        self.callback_2 = ModelCheckpoint(filepath=self.model_path, save_weights_only=True, verbose=1)
        self.callback_3 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor= 0.5, min_lr=0.00001)
        
        self.train_model()
        self.evaluate_model()
        self.plot_model()
        self.plot_random_examples()


    def setup_structure(self):
        
        if self.image_type == "normal":
            self.true_path = self.true_path + "brain_cancer_seperate_category_2/"
        elif self.image_type == "edge_1":
            self.true_path = self.true_path + "brain_cancer_seperate_category_2_edge_1/"
        elif self.image_type == "edge_2":
            self.true_path = self.true_path + "brain_cancer_seperate_category_2_edge_2/"

        if self.number_classes == 2:

            self.category_names = self.categories 
            
            self.check_valid(self.categories[0])
            self.check_valid(self.categories[1])
            
            self.resize_image_and_label_image(self.categories[0])
            self.resize_image_and_label_image(self.categories[1])

        elif self.number_classes == 4:
            
            self.category_names = self.advanced_categories 

            self.true_path = "brain_cancer_category_4/"
            if self.image_type == "normal":
            	self.true_path = self.true_path + "brain_cancer_seperate_category_4/"
            elif self.image_type == "edge_1":
                self.true_path = self.true_path + "brain_cancer_seperate_category_4_edge_1/"
            elif self.image_type == "edge_2":
                self.true_path = self.true_path + "brain_cancer_seperate_category_4_edge_2/"
             
            self.check_valid(self.advanced_categories[0])
            self.check_valid(self.advanced_categories[1])
            self.check_valid(self.advanced_categories[2])
            self.check_valid(self.advanced_categories[3])
            
            self.resize_image_and_label_image(self.advanced_categories[0])
            self.resize_image_and_label_image(self.advanced_categories[1])
            self.resize_image_and_label_image(self.advanced_categories[2])
            self.resize_image_and_label_image(self.advanced_categories[3])

        else:
            print("Detection Variety out of bounds")


    def check_valid(self, input_file):
        for img in os.listdir(self.true_path + input_file):
            ext = os.path.splitext(img)[1]
            if ext.lower() not in self.valid_images:
                continue
    

    def resize_image_and_label_image(self, input_file):
        for image in os.listdir(self.true_path + input_file):
            
            image_resized = cv2.imread(os.path.join(self.true_path + input_file,image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            self.image_file.append(image_resized)

            if input_file == "False":
                self.label_name.append(0)
            elif input_file == "True":
                self.label_name.append(1)
            elif input_file == "glioma_tumor":
                self.label_name.append(1)
            elif input_file == "meningioma_tumor":
                self.label_name.append(2)
            elif input_file == "pituitary_tumor":
                self.label_name.append(3)
            else:
                print("error")


    def splitting_data_normalize(self):
        self.X_train, self.X_test, self.Y_train_vec, self.Y_test_vec = train_test_split(self.image_file, self.label_name, test_size = 0.10, random_state = 42)
        self.input_shape = self.X_train.shape[1:]
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train_vec, self.number_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test_vec, self.number_classes)
        self.X_train = self.X_train.astype("float32") /255
        self.X_test = self.X_test.astype("float32") / 255


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
